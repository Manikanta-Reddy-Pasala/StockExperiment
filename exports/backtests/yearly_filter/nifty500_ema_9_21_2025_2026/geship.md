# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1589.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 25 |
| ALERT3 | 109 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 70 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 18 / 54
- **Target hits / Stop hits / Partials:** 1 / 69 / 2
- **Avg / median % per leg:** -0.71% / -0.72%
- **Sum % (uncompounded):** -51.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 10 | 28.6% | 1 | 33 | 1 | 0.21% | 7.3% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.90% | 11.7% |
| BUY @ 3rd Alert (retest2) | 32 | 7 | 21.9% | 1 | 31 | 0 | -0.14% | -4.4% |
| SELL (all) | 37 | 8 | 21.6% | 0 | 36 | 1 | -1.58% | -58.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 8 | 21.6% | 0 | 36 | 1 | -1.58% | -58.6% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.90% | 11.7% |
| retest2 (combined) | 69 | 15 | 21.7% | 1 | 67 | 1 | -0.91% | -63.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 907.55 | 883.80 | 880.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 916.05 | 905.27 | 898.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 12:15:00 | 911.00 | 914.29 | 908.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:45:00 | 911.25 | 914.29 | 908.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 911.00 | 912.98 | 909.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 915.00 | 912.98 | 909.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 924.10 | 942.41 | 944.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 924.10 | 942.41 | 944.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 917.50 | 937.43 | 941.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 12:15:00 | 915.35 | 915.03 | 922.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:30:00 | 916.00 | 915.03 | 922.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 925.30 | 915.95 | 920.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 926.95 | 915.95 | 920.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 928.40 | 918.44 | 921.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 928.40 | 918.44 | 921.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 926.00 | 923.15 | 922.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 929.50 | 924.42 | 923.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 926.05 | 930.00 | 927.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 935.55 | 932.57 | 930.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 936.85 | 934.46 | 931.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 926.50 | 935.66 | 936.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 926.50 | 935.66 | 936.30 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 965.55 | 939.17 | 937.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 987.65 | 975.55 | 964.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 991.05 | 995.93 | 988.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 991.05 | 995.93 | 988.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 988.50 | 994.44 | 988.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 989.80 | 994.44 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 988.10 | 993.18 | 988.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 995.00 | 989.75 | 988.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 986.40 | 987.79 | 987.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 986.40 | 987.79 | 987.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 978.75 | 985.98 | 986.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 983.30 | 982.25 | 984.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 983.30 | 982.25 | 984.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 971.10 | 980.05 | 983.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:15:00 | 969.50 | 980.05 | 983.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 967.00 | 976.57 | 980.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 964.90 | 974.24 | 979.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 968.15 | 970.99 | 976.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 975.00 | 971.79 | 976.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:30:00 | 972.90 | 971.79 | 976.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 976.25 | 972.68 | 976.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:30:00 | 974.10 | 973.67 | 976.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 968.35 | 973.67 | 976.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 956.75 | 974.47 | 976.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1004.05 | 980.38 | 978.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 09:15:00 | 1004.05 | 980.38 | 978.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 10:15:00 | 1033.95 | 991.10 | 983.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 997.00 | 1004.87 | 992.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 999.10 | 1004.87 | 992.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 989.85 | 1001.87 | 992.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 993.20 | 1001.87 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 985.00 | 998.49 | 991.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1009.50 | 998.49 | 991.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 979.90 | 997.24 | 995.32 | SL hit (close<static) qty=1.00 sl=984.10 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 986.00 | 993.01 | 993.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 975.00 | 989.41 | 991.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 961.55 | 956.85 | 965.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 961.55 | 956.85 | 965.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 966.70 | 958.82 | 965.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 966.70 | 958.82 | 965.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 966.30 | 960.32 | 965.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 967.25 | 960.32 | 965.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 970.30 | 962.31 | 965.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 970.30 | 962.31 | 965.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 959.75 | 961.80 | 965.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 958.00 | 961.80 | 965.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 973.00 | 963.43 | 965.31 | SL hit (close>static) qty=1.00 sl=972.55 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 976.50 | 967.10 | 966.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 991.80 | 972.94 | 969.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 972.00 | 978.82 | 973.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 973.05 | 978.82 | 973.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 977.30 | 978.52 | 973.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 983.20 | 975.06 | 973.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:45:00 | 981.05 | 984.12 | 979.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 981.55 | 982.63 | 981.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 976.90 | 981.83 | 981.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 976.90 | 981.83 | 981.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 973.80 | 980.22 | 981.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 986.90 | 980.84 | 981.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 991.40 | 980.84 | 981.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 984.80 | 981.63 | 981.59 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 979.10 | 981.53 | 981.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 13:15:00 | 978.00 | 980.82 | 981.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 981.60 | 980.59 | 981.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 981.00 | 980.59 | 981.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 986.35 | 981.74 | 981.49 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 973.55 | 980.94 | 981.61 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 985.30 | 982.12 | 982.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 999.60 | 986.73 | 984.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 1002.00 | 1002.43 | 994.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1015.00 | 1001.93 | 995.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1019.55 | 1023.62 | 1019.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 1015.60 | 1021.15 | 1019.38 | SL hit (close<ema400) qty=1.00 sl=1019.38 alert=retest1 |

### Cycle 16 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 1009.10 | 1017.59 | 1018.00 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 1020.70 | 1018.49 | 1018.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 1029.30 | 1021.17 | 1019.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1015.20 | 1020.92 | 1019.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1015.20 | 1020.92 | 1019.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1011.50 | 1019.04 | 1019.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 1007.95 | 1015.65 | 1017.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1001.20 | 1000.25 | 1006.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1001.20 | 1000.25 | 1006.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1000.25 | 994.94 | 999.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:00:00 | 987.95 | 997.71 | 999.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:45:00 | 990.10 | 992.52 | 994.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 992.85 | 993.78 | 994.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 992.45 | 992.68 | 993.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 991.65 | 991.12 | 992.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 992.65 | 991.12 | 992.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 982.50 | 989.30 | 991.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 991.55 | 990.56 | 990.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 991.55 | 990.56 | 990.53 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 988.60 | 990.34 | 990.45 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 993.90 | 990.97 | 990.72 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 985.15 | 989.81 | 990.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 981.10 | 987.42 | 989.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 964.00 | 962.75 | 972.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 14:30:00 | 961.85 | 962.75 | 972.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 959.00 | 939.65 | 948.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 959.00 | 939.65 | 948.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 957.05 | 943.13 | 948.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:15:00 | 951.40 | 943.13 | 948.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 942.00 | 930.17 | 929.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 942.00 | 930.17 | 929.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 945.45 | 935.36 | 932.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 961.25 | 962.24 | 953.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 12:00:00 | 961.25 | 962.24 | 953.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 956.00 | 960.97 | 955.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 15:00:00 | 972.00 | 958.03 | 955.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 967.45 | 961.76 | 958.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 971.60 | 975.94 | 976.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 971.60 | 975.94 | 976.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 969.60 | 974.20 | 975.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 940.95 | 937.37 | 944.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 940.95 | 937.37 | 944.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 946.85 | 936.43 | 938.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 946.85 | 936.43 | 938.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 955.90 | 940.32 | 940.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 961.35 | 946.96 | 943.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 958.00 | 959.52 | 952.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 958.35 | 959.52 | 952.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 966.45 | 960.91 | 954.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 969.50 | 960.91 | 954.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 971.50 | 962.23 | 958.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 970.50 | 969.34 | 964.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 953.85 | 963.78 | 964.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 953.85 | 963.78 | 964.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 952.50 | 961.53 | 963.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 979.00 | 963.66 | 964.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 983.70 | 963.66 | 964.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 978.90 | 966.71 | 965.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 12:15:00 | 989.05 | 973.46 | 968.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 15:15:00 | 975.00 | 975.46 | 971.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:15:00 | 986.70 | 975.46 | 971.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 986.40 | 987.12 | 984.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:30:00 | 992.20 | 988.83 | 985.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:00:00 | 992.75 | 989.84 | 986.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 12:15:00 | 1036.04 | 1017.15 | 1004.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 1052.15 | 1057.48 | 1048.30 | SL hit (close<ema200) qty=0.50 sl=1057.48 alert=retest1 |

### Cycle 28 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 1039.75 | 1049.05 | 1050.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 1024.70 | 1039.79 | 1044.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1002.95 | 1001.24 | 1012.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 1002.85 | 1001.24 | 1012.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1002.40 | 994.33 | 1002.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 1013.05 | 994.33 | 1002.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 993.30 | 994.13 | 1002.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 990.00 | 994.13 | 1002.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 986.15 | 991.48 | 996.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1034.40 | 1000.72 | 998.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1034.40 | 1000.72 | 998.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1043.45 | 1009.27 | 1002.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1050.85 | 1054.02 | 1041.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:45:00 | 1049.00 | 1054.02 | 1041.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1057.80 | 1051.80 | 1045.72 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1046.75 | 1048.99 | 1049.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1044.85 | 1046.99 | 1048.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 1034.80 | 1031.93 | 1037.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 1034.80 | 1031.93 | 1037.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1034.55 | 1032.73 | 1036.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 1038.15 | 1032.73 | 1036.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1038.05 | 1033.68 | 1036.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 1041.15 | 1033.68 | 1036.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1034.65 | 1033.87 | 1036.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 1033.75 | 1033.96 | 1035.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 12:15:00 | 1040.60 | 1035.29 | 1036.39 | SL hit (close>static) qty=1.00 sl=1038.65 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 1045.05 | 1038.26 | 1037.61 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1028.85 | 1037.17 | 1037.54 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1036.45 | 1035.52 | 1035.42 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 1032.40 | 1034.80 | 1035.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 12:15:00 | 1029.80 | 1033.22 | 1034.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1032.30 | 1027.88 | 1030.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 1032.30 | 1027.88 | 1030.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1016.00 | 1025.51 | 1029.59 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 1040.90 | 1030.08 | 1029.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 1062.00 | 1045.99 | 1038.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1053.10 | 1054.35 | 1047.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1061.75 | 1055.55 | 1048.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1074.00 | 1083.23 | 1083.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1074.00 | 1083.23 | 1083.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1068.80 | 1080.34 | 1082.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1059.00 | 1054.76 | 1062.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1059.00 | 1054.76 | 1062.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1071.90 | 1058.19 | 1063.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1070.50 | 1058.19 | 1063.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1076.50 | 1061.85 | 1064.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 1076.70 | 1061.85 | 1064.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1081.50 | 1068.83 | 1067.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 1090.50 | 1076.26 | 1071.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1147.00 | 1148.30 | 1128.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 1142.20 | 1148.30 | 1128.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1127.00 | 1140.53 | 1129.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1127.30 | 1140.53 | 1129.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1127.10 | 1137.84 | 1129.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 1135.60 | 1137.84 | 1129.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 1132.10 | 1135.80 | 1130.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1133.00 | 1135.36 | 1130.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1135.70 | 1135.43 | 1131.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1136.50 | 1135.48 | 1131.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1134.00 | 1135.48 | 1131.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1133.00 | 1134.84 | 1132.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1139.30 | 1134.84 | 1132.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 1123.90 | 1132.35 | 1131.70 | SL hit (close<static) qty=1.00 sl=1125.60 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1115.00 | 1128.88 | 1130.18 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1132.80 | 1129.99 | 1129.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 15:15:00 | 1137.50 | 1131.61 | 1130.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1126.70 | 1130.62 | 1130.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 1122.80 | 1130.62 | 1130.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1122.50 | 1129.00 | 1129.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 1117.70 | 1124.60 | 1127.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 1107.50 | 1106.95 | 1114.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 11:45:00 | 1106.10 | 1106.95 | 1114.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1076.00 | 1072.83 | 1083.30 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1095.80 | 1086.32 | 1086.25 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1084.30 | 1085.91 | 1086.07 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 11:15:00 | 1092.30 | 1086.37 | 1086.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 1094.90 | 1088.67 | 1087.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1104.00 | 1115.77 | 1108.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1104.00 | 1115.77 | 1108.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1104.80 | 1113.57 | 1108.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 1098.50 | 1113.57 | 1108.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 13:15:00 | 1096.00 | 1105.18 | 1105.23 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 1110.90 | 1104.26 | 1104.14 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 1096.50 | 1103.66 | 1104.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 11:15:00 | 1086.40 | 1098.36 | 1101.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1075.30 | 1074.17 | 1083.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 1075.30 | 1074.17 | 1083.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1080.20 | 1076.26 | 1080.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:30:00 | 1082.80 | 1076.26 | 1080.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1081.60 | 1077.33 | 1080.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 1081.60 | 1077.33 | 1080.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 1089.90 | 1079.84 | 1081.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 1089.90 | 1079.84 | 1081.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1085.60 | 1081.00 | 1081.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1081.00 | 1081.00 | 1081.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 1086.10 | 1082.02 | 1081.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 1086.10 | 1082.02 | 1081.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 1091.00 | 1083.65 | 1082.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1120.00 | 1120.39 | 1112.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 1120.00 | 1120.39 | 1112.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1111.90 | 1118.69 | 1112.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1111.90 | 1118.69 | 1112.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1109.60 | 1116.87 | 1111.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 1107.60 | 1116.87 | 1111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1107.60 | 1115.02 | 1111.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 1107.60 | 1115.02 | 1111.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1108.50 | 1113.71 | 1111.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1111.80 | 1110.91 | 1110.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 1112.30 | 1111.16 | 1110.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 1110.60 | 1110.67 | 1110.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:30:00 | 1111.90 | 1110.81 | 1110.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1111.40 | 1110.93 | 1110.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:30:00 | 1109.90 | 1110.93 | 1110.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1112.10 | 1111.16 | 1110.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:15:00 | 1106.00 | 1111.16 | 1110.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 1106.00 | 1110.13 | 1110.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 1106.00 | 1110.13 | 1110.24 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 09:15:00 | 1118.10 | 1111.72 | 1110.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 1130.00 | 1116.75 | 1114.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1113.90 | 1119.91 | 1116.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1113.90 | 1119.91 | 1116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 14:15:00 | 1094.20 | 1114.77 | 1114.79 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 15:15:00 | 1111.80 | 1108.90 | 1108.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1114.10 | 1109.94 | 1109.04 | Break + close above crossover candle high |

### Cycle 52 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1098.10 | 1107.57 | 1108.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1097.40 | 1103.26 | 1105.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1108.80 | 1102.53 | 1104.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 1108.20 | 1102.53 | 1104.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1102.40 | 1102.51 | 1104.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 1100.50 | 1103.11 | 1104.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1100.20 | 1103.26 | 1104.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 1099.10 | 1102.13 | 1103.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:15:00 | 1100.20 | 1102.13 | 1103.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1102.00 | 1102.11 | 1103.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1102.00 | 1102.11 | 1103.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1102.50 | 1102.19 | 1103.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1107.20 | 1102.19 | 1103.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1121.10 | 1105.97 | 1104.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 1121.10 | 1105.97 | 1104.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 1132.70 | 1114.04 | 1108.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1118.20 | 1123.10 | 1116.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 1117.10 | 1123.10 | 1116.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1115.50 | 1121.58 | 1116.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 1115.50 | 1121.58 | 1116.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1118.00 | 1120.86 | 1116.74 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1110.10 | 1115.17 | 1115.39 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1126.50 | 1116.77 | 1115.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1134.00 | 1122.25 | 1118.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1109.00 | 1121.77 | 1120.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1099.50 | 1121.77 | 1120.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1108.10 | 1119.03 | 1119.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 1104.20 | 1116.07 | 1117.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1105.90 | 1105.10 | 1110.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:45:00 | 1106.20 | 1105.10 | 1110.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1106.90 | 1105.08 | 1108.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1106.90 | 1105.08 | 1108.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1112.00 | 1106.51 | 1108.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 1102.80 | 1105.19 | 1107.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 1102.80 | 1095.81 | 1100.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 1101.70 | 1093.10 | 1093.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 1101.60 | 1094.80 | 1093.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1101.60 | 1094.80 | 1093.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 13:15:00 | 1107.00 | 1098.99 | 1096.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1105.90 | 1113.38 | 1108.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1104.50 | 1113.38 | 1108.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1111.00 | 1112.90 | 1108.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 1117.20 | 1114.38 | 1110.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1103.90 | 1112.46 | 1110.10 | SL hit (close<static) qty=1.00 sl=1106.10 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1102.20 | 1108.62 | 1108.66 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 14:15:00 | 1116.30 | 1109.02 | 1108.70 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1099.70 | 1107.20 | 1108.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1095.50 | 1104.86 | 1106.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1095.20 | 1093.20 | 1099.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 1097.10 | 1093.20 | 1099.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 1082.00 | 1090.96 | 1097.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:30:00 | 1093.00 | 1090.96 | 1097.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1099.00 | 1093.39 | 1097.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1099.00 | 1093.39 | 1097.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1092.60 | 1093.23 | 1097.29 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1111.80 | 1101.07 | 1099.86 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1095.60 | 1104.39 | 1105.33 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 14:15:00 | 1115.50 | 1107.46 | 1106.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 1122.00 | 1110.37 | 1107.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 10:15:00 | 1196.20 | 1196.27 | 1182.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:30:00 | 1196.00 | 1196.27 | 1182.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1170.50 | 1191.12 | 1181.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1170.50 | 1191.12 | 1181.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1190.80 | 1191.05 | 1182.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1169.30 | 1191.05 | 1182.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1174.40 | 1188.41 | 1183.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1220.40 | 1188.41 | 1183.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-10 09:15:00 | 1342.44 | 1312.00 | 1290.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1300.00 | 1327.49 | 1329.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 12:15:00 | 1289.70 | 1305.65 | 1312.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1305.20 | 1299.81 | 1306.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 1303.50 | 1299.81 | 1306.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1304.80 | 1300.81 | 1306.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1304.80 | 1300.81 | 1306.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1297.80 | 1300.21 | 1305.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 1297.50 | 1299.67 | 1305.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:00:00 | 1297.40 | 1299.21 | 1304.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1294.20 | 1296.92 | 1301.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 12:15:00 | 1290.80 | 1297.54 | 1301.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1306.60 | 1294.36 | 1297.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1309.50 | 1294.36 | 1297.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1308.40 | 1297.17 | 1298.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 1299.00 | 1297.58 | 1298.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1296.10 | 1298.92 | 1298.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 1300.70 | 1299.28 | 1299.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 1300.70 | 1299.28 | 1299.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 1303.30 | 1300.08 | 1299.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1341.40 | 1342.79 | 1331.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 1341.40 | 1342.79 | 1331.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1339.40 | 1343.96 | 1336.80 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1328.60 | 1335.34 | 1336.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 1318.80 | 1332.03 | 1334.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 10:15:00 | 1328.10 | 1326.61 | 1330.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 11:00:00 | 1328.10 | 1326.61 | 1330.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1329.50 | 1327.18 | 1330.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:15:00 | 1324.20 | 1327.18 | 1330.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 13:15:00 | 1344.00 | 1330.74 | 1331.67 | SL hit (close>static) qty=1.00 sl=1338.60 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1354.90 | 1335.57 | 1333.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 1417.50 | 1354.20 | 1342.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 1409.00 | 1410.27 | 1391.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 1381.40 | 1410.27 | 1391.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1386.50 | 1405.52 | 1390.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 1386.10 | 1405.52 | 1390.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 1382.90 | 1400.99 | 1389.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 1391.60 | 1400.99 | 1389.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1362.50 | 1393.30 | 1387.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 12:00:00 | 1362.50 | 1393.30 | 1387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1335.50 | 1374.82 | 1379.63 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1389.30 | 1378.63 | 1377.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1441.00 | 1395.00 | 1385.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 1455.00 | 1460.91 | 1439.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 1455.00 | 1460.91 | 1439.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1450.00 | 1458.56 | 1446.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 1396.90 | 1458.56 | 1446.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1380.00 | 1442.85 | 1440.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 1380.00 | 1442.85 | 1440.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 1348.20 | 1423.92 | 1432.33 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1468.70 | 1422.13 | 1417.20 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 15:15:00 | 1428.10 | 1432.68 | 1432.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1370.70 | 1420.29 | 1427.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1391.00 | 1380.91 | 1398.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 1404.50 | 1380.91 | 1398.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1390.60 | 1382.85 | 1397.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:30:00 | 1385.40 | 1386.08 | 1396.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 1383.80 | 1386.55 | 1396.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1454.90 | 1402.25 | 1401.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1454.90 | 1402.25 | 1401.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1456.30 | 1413.06 | 1406.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 1413.60 | 1418.88 | 1411.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:45:00 | 1420.20 | 1418.88 | 1411.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1420.00 | 1419.10 | 1412.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1434.90 | 1419.10 | 1412.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1427.00 | 1420.68 | 1413.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1469.00 | 1418.29 | 1417.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 1444.00 | 1444.55 | 1438.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 1445.40 | 1443.64 | 1438.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1470.70 | 1440.68 | 1438.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1487.40 | 1450.03 | 1443.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 1430.00 | 1446.28 | 1447.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 1430.00 | 1446.28 | 1447.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 12:15:00 | 1427.90 | 1442.61 | 1445.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1433.80 | 1431.88 | 1438.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:30:00 | 1425.30 | 1423.65 | 1434.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:15:00 | 1354.03 | 1373.57 | 1386.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1396.40 | 1358.50 | 1369.53 | SL hit (close>ema200) qty=0.50 sl=1358.50 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 1399.10 | 1377.35 | 1376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1400.20 | 1385.45 | 1380.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 1382.50 | 1388.23 | 1383.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 1382.50 | 1388.23 | 1383.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1385.30 | 1387.64 | 1383.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 1381.50 | 1387.64 | 1383.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1377.20 | 1385.55 | 1383.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:00:00 | 1377.20 | 1385.55 | 1383.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 1380.80 | 1384.60 | 1383.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1397.50 | 1384.22 | 1383.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 13:15:00 | 1421.00 | 1434.41 | 1435.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1421.00 | 1434.41 | 1435.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1412.50 | 1430.03 | 1433.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1416.80 | 1413.18 | 1420.99 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1467.80 | 1430.76 | 1426.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1523.80 | 1471.82 | 1451.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1526.70 | 1556.87 | 1532.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 1526.70 | 1556.87 | 1532.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1546.40 | 1554.77 | 1533.60 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1526.00 | 1530.53 | 1531.14 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 1553.00 | 1533.07 | 1531.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 1566.30 | 1539.72 | 1535.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1549.30 | 1551.85 | 1542.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 1551.00 | 1551.85 | 1542.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1583.10 | 1560.28 | 1551.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1610.00 | 1560.28 | 1551.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 1589.10 | 1580.66 | 1566.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:30:00 | 868.80 | 2025-05-12 10:15:00 | 900.90 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2025-05-16 09:15:00 | 915.00 | 2025-05-20 13:15:00 | 924.10 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-05-28 09:30:00 | 935.55 | 2025-05-30 10:15:00 | 926.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-28 10:45:00 | 936.85 | 2025-05-30 10:15:00 | 926.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-06-09 09:30:00 | 995.00 | 2025-06-09 15:15:00 | 986.40 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-11 10:15:00 | 969.50 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-06-11 13:15:00 | 967.00 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-06-11 14:00:00 | 964.90 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-06-12 09:30:00 | 968.15 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-06-12 13:30:00 | 974.10 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-06-12 14:15:00 | 968.35 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2025-06-13 09:15:00 | 956.75 | 2025-06-13 09:15:00 | 1004.05 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2025-06-16 09:15:00 | 1009.50 | 2025-06-17 09:15:00 | 979.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-06-20 15:15:00 | 958.00 | 2025-06-23 09:15:00 | 973.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-25 09:15:00 | 983.20 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-25 13:45:00 | 981.05 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-26 12:45:00 | 981.55 | 2025-06-27 13:15:00 | 976.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-07-04 09:15:00 | 1015.00 | 2025-07-08 14:15:00 | 1015.60 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-15 15:00:00 | 987.95 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-17 11:45:00 | 990.10 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-17 15:15:00 | 992.85 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-07-18 09:30:00 | 992.45 | 2025-07-23 10:15:00 | 991.55 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-07-30 09:15:00 | 951.40 | 2025-08-07 14:15:00 | 942.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-08-14 15:00:00 | 972.00 | 2025-08-22 11:15:00 | 971.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-18 10:30:00 | 967.45 | 2025-08-22 11:15:00 | 971.60 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-09-03 10:15:00 | 969.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-09-04 09:30:00 | 971.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-09-05 09:45:00 | 970.50 | 2025-09-08 13:15:00 | 953.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-09-10 09:15:00 | 986.70 | 2025-09-16 12:15:00 | 1036.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-10 09:15:00 | 986.70 | 2025-09-19 10:15:00 | 1052.15 | STOP_HIT | 0.50 | 6.63% |
| BUY | retest2 | 2025-09-12 12:30:00 | 992.20 | 2025-09-23 11:15:00 | 1039.75 | STOP_HIT | 1.00 | 4.79% |
| BUY | retest2 | 2025-09-12 15:00:00 | 992.75 | 2025-09-23 11:15:00 | 1039.75 | STOP_HIT | 1.00 | 4.73% |
| SELL | retest2 | 2025-09-30 11:15:00 | 990.00 | 2025-10-03 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-10-01 10:45:00 | 986.15 | 2025-10-03 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-10-16 11:30:00 | 1033.75 | 2025-10-16 12:15:00 | 1040.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-10-29 10:45:00 | 1061.75 | 2025-11-04 10:15:00 | 1074.00 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-11-13 14:15:00 | 1135.60 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-14 10:15:00 | 1132.10 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1133.00 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1135.70 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-17 09:15:00 | 1139.30 | 2025-11-17 11:15:00 | 1123.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1081.00 | 2025-12-10 14:15:00 | 1086.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1111.80 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-17 09:45:00 | 1112.30 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 11:30:00 | 1110.60 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-12-17 12:30:00 | 1111.90 | 2025-12-17 15:15:00 | 1106.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-29 11:30:00 | 1100.50 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-29 15:15:00 | 1100.20 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-12-30 13:45:00 | 1099.10 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-12-30 14:15:00 | 1100.20 | 2025-12-31 09:15:00 | 1121.10 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1102.80 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-09 10:45:00 | 1102.80 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-01-13 10:15:00 | 1101.70 | 2026-01-13 10:15:00 | 1101.60 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-01-16 14:30:00 | 1117.20 | 2026-01-19 09:15:00 | 1103.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1220.40 | 2026-02-10 09:15:00 | 1342.44 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-19 13:00:00 | 1297.50 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1297.40 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-02-20 10:30:00 | 1294.20 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-02-20 12:15:00 | 1290.80 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-02-23 11:45:00 | 1299.00 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-02-24 09:45:00 | 1296.10 | 2026-02-24 10:15:00 | 1300.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1324.20 | 2026-03-04 13:15:00 | 1344.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1323.70 | 2026-03-04 14:15:00 | 1354.90 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-24 12:30:00 | 1385.40 | 2026-03-25 09:15:00 | 1454.90 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2026-03-24 13:45:00 | 1383.80 | 2026-03-25 09:15:00 | 1454.90 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1469.00 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-04-02 11:45:00 | 1444.00 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-04-02 12:30:00 | 1445.40 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-06 09:15:00 | 1470.70 | 2026-04-07 11:15:00 | 1430.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-08 10:30:00 | 1425.30 | 2026-04-13 09:15:00 | 1354.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-08 10:30:00 | 1425.30 | 2026-04-15 09:15:00 | 1396.40 | STOP_HIT | 0.50 | 2.03% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1397.50 | 2026-04-23 13:15:00 | 1421.00 | STOP_HIT | 1.00 | 1.68% |
