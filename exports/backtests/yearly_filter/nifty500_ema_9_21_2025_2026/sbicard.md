# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 645.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 19 |
| ALERT3 | 113 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 42
- **Target hits / Stop hits / Partials:** 3 / 53 / 7
- **Avg / median % per leg:** 0.91% / -0.33%
- **Sum % (uncompounded):** 57.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 6 | 20.0% | 0 | 30 | 0 | -0.48% | -14.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 6 | 20.0% | 0 | 30 | 0 | -0.48% | -14.3% |
| SELL (all) | 33 | 15 | 45.5% | 3 | 23 | 7 | 2.16% | 71.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 15 | 45.5% | 3 | 23 | 7 | 2.16% | 71.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 21 | 33.3% | 3 | 53 | 7 | 0.91% | 57.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 900.65 | 892.11 | 891.48 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 889.35 | 892.08 | 892.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 15:15:00 | 887.90 | 890.15 | 891.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 898.20 | 891.76 | 891.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:00:00 | 898.20 | 891.76 | 891.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 899.00 | 893.21 | 892.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 899.95 | 894.56 | 893.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 910.10 | 911.30 | 907.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 910.10 | 911.30 | 907.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 912.60 | 912.99 | 910.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 908.75 | 912.99 | 910.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 904.70 | 911.06 | 909.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 905.20 | 911.06 | 909.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 903.55 | 909.56 | 909.22 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 901.45 | 907.94 | 908.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 898.70 | 904.76 | 906.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 903.70 | 903.21 | 905.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 13:00:00 | 903.70 | 903.21 | 905.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 904.15 | 903.40 | 905.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:30:00 | 903.60 | 903.40 | 905.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 895.30 | 892.17 | 896.92 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 911.10 | 899.46 | 898.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 914.00 | 902.37 | 900.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 900.55 | 904.95 | 902.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 900.55 | 904.95 | 902.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 910.90 | 906.14 | 903.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 914.15 | 907.74 | 904.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 913.30 | 909.88 | 906.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:45:00 | 913.25 | 911.22 | 910.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 14:15:00 | 912.30 | 911.21 | 910.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 914.20 | 911.81 | 910.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 916.50 | 913.58 | 911.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 916.80 | 918.86 | 918.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 910.85 | 917.26 | 917.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 910.85 | 917.26 | 917.91 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 935.25 | 920.49 | 919.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 943.30 | 925.05 | 921.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 1002.90 | 1006.09 | 990.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:00:00 | 1002.90 | 1006.09 | 990.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 993.50 | 999.47 | 995.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 993.00 | 999.47 | 995.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 992.30 | 998.04 | 995.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 992.30 | 998.04 | 995.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 993.20 | 1000.48 | 998.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 996.95 | 999.53 | 998.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 995.85 | 998.37 | 998.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 994.25 | 1002.27 | 1002.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 994.25 | 1002.27 | 1002.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 991.60 | 997.76 | 1000.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 944.60 | 943.12 | 954.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 944.60 | 943.12 | 954.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 948.60 | 944.71 | 953.54 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 974.15 | 956.80 | 956.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 979.60 | 961.36 | 958.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 974.40 | 976.72 | 971.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 14:00:00 | 974.40 | 976.72 | 971.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 972.95 | 975.97 | 971.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 970.10 | 975.97 | 971.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 961.00 | 972.98 | 970.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 963.40 | 972.98 | 970.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 953.25 | 969.03 | 968.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 953.25 | 969.03 | 968.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 10:15:00 | 958.20 | 966.86 | 967.93 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 977.00 | 969.30 | 968.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 982.65 | 971.97 | 970.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 957.65 | 969.87 | 969.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 957.65 | 969.87 | 969.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 982.40 | 972.38 | 970.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:00:00 | 988.35 | 975.77 | 972.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 960.30 | 970.32 | 971.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 960.30 | 970.32 | 971.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 14:15:00 | 957.20 | 967.70 | 970.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 914.70 | 913.12 | 923.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:30:00 | 915.70 | 913.12 | 923.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 914.00 | 911.89 | 916.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 917.55 | 911.89 | 916.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 913.85 | 910.26 | 912.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 913.85 | 910.26 | 912.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 915.50 | 911.31 | 912.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 916.25 | 911.31 | 912.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 920.30 | 914.83 | 914.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 924.65 | 916.79 | 915.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 931.70 | 932.08 | 926.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 931.70 | 932.08 | 926.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 923.85 | 931.60 | 928.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 922.00 | 931.60 | 928.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 920.30 | 929.34 | 927.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 920.30 | 929.34 | 927.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 919.50 | 925.80 | 926.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 915.00 | 923.64 | 925.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 14:15:00 | 898.00 | 896.15 | 900.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:45:00 | 899.00 | 896.15 | 900.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 898.10 | 896.51 | 900.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 902.40 | 896.51 | 900.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 901.40 | 897.49 | 900.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 901.40 | 897.49 | 900.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 901.50 | 898.29 | 900.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 903.00 | 898.29 | 900.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 905.00 | 900.55 | 901.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:45:00 | 905.15 | 900.55 | 901.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 14:15:00 | 910.20 | 902.48 | 901.89 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 898.15 | 901.27 | 901.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 894.00 | 898.48 | 899.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 899.55 | 897.52 | 899.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 899.80 | 897.52 | 899.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 899.25 | 897.87 | 899.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:00:00 | 896.10 | 897.77 | 898.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 895.40 | 898.47 | 898.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 15:15:00 | 895.50 | 898.37 | 898.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 851.29 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 850.63 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 850.72 | 879.91 | 885.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-31 09:15:00 | 806.49 | 815.94 | 828.94 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 798.40 | 794.92 | 794.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 800.00 | 796.70 | 795.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 794.25 | 796.48 | 795.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 794.25 | 796.48 | 795.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 792.35 | 795.66 | 795.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 792.35 | 795.66 | 795.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 13:15:00 | 788.55 | 794.23 | 794.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 787.05 | 791.32 | 793.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 790.60 | 790.33 | 792.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:45:00 | 790.85 | 790.33 | 792.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 790.45 | 790.36 | 791.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 792.70 | 790.36 | 791.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 790.35 | 790.06 | 791.51 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 808.45 | 793.19 | 792.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 812.00 | 799.56 | 795.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 815.55 | 816.18 | 810.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:45:00 | 816.00 | 816.18 | 810.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 820.05 | 823.47 | 821.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 818.05 | 823.47 | 821.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 826.00 | 823.98 | 821.62 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 808.70 | 820.05 | 820.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 803.85 | 816.81 | 819.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 815.65 | 813.85 | 816.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 815.65 | 813.85 | 816.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 802.65 | 802.20 | 805.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 802.65 | 802.20 | 805.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 804.10 | 802.58 | 805.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 800.25 | 802.58 | 805.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 800.70 | 802.28 | 805.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 806.05 | 802.21 | 804.25 | SL hit (close>static) qty=1.00 sl=805.90 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 815.35 | 807.12 | 806.19 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 799.55 | 804.79 | 805.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 795.80 | 802.99 | 804.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 803.80 | 800.58 | 802.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 803.80 | 800.58 | 802.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 797.85 | 800.03 | 802.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 796.20 | 800.03 | 802.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:30:00 | 795.10 | 798.32 | 801.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 805.40 | 798.46 | 800.11 | SL hit (close>static) qty=1.00 sl=803.90 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 801.45 | 796.68 | 796.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 14:15:00 | 803.00 | 798.64 | 797.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 15:15:00 | 853.50 | 856.14 | 848.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:15:00 | 860.20 | 856.14 | 848.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 866.15 | 858.14 | 850.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:45:00 | 874.40 | 864.08 | 855.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 880.90 | 886.37 | 886.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 880.90 | 886.37 | 886.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 877.35 | 884.56 | 885.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 872.85 | 871.92 | 876.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:30:00 | 872.55 | 871.92 | 876.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 878.75 | 872.17 | 873.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 880.95 | 872.17 | 873.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 880.30 | 873.80 | 874.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 880.30 | 873.80 | 874.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 11:15:00 | 880.50 | 875.14 | 874.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 12:15:00 | 886.30 | 877.37 | 876.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 15:15:00 | 880.00 | 880.18 | 877.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:15:00 | 882.10 | 880.18 | 877.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 886.00 | 881.34 | 878.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 889.30 | 883.81 | 879.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 869.10 | 882.85 | 881.71 | SL hit (close<static) qty=1.00 sl=875.30 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 878.80 | 880.98 | 880.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 872.90 | 877.00 | 878.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 878.70 | 877.34 | 878.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 878.70 | 877.34 | 878.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 874.80 | 876.83 | 878.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 873.20 | 876.83 | 878.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 874.15 | 871.33 | 874.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 872.65 | 871.60 | 873.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 872.65 | 871.81 | 873.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 872.90 | 872.02 | 873.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 868.20 | 873.97 | 874.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 875.70 | 867.37 | 869.74 | SL hit (close>static) qty=1.00 sl=874.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 875.55 | 871.60 | 871.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 882.50 | 873.78 | 872.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 904.65 | 905.25 | 897.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 904.65 | 905.25 | 897.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 921.30 | 920.25 | 912.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 923.05 | 919.92 | 914.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 13:45:00 | 924.00 | 920.67 | 917.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 924.95 | 921.53 | 918.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 922.45 | 920.95 | 918.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 922.00 | 921.88 | 919.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:30:00 | 923.25 | 921.88 | 919.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 920.95 | 921.69 | 919.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 920.95 | 921.69 | 919.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 920.05 | 921.36 | 919.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 918.65 | 920.81 | 919.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 915.45 | 919.74 | 919.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 915.45 | 919.74 | 919.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 914.55 | 918.70 | 918.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 914.55 | 918.70 | 918.95 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 925.70 | 919.51 | 918.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 933.35 | 922.28 | 920.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 15:15:00 | 935.00 | 935.51 | 930.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:15:00 | 937.65 | 935.51 | 930.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 929.65 | 933.69 | 931.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 929.65 | 933.69 | 931.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 929.85 | 932.92 | 931.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 926.90 | 932.92 | 931.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 928.00 | 930.96 | 930.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 930.55 | 930.96 | 930.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 15:15:00 | 927.90 | 936.11 | 936.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 927.90 | 936.11 | 936.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 922.50 | 930.40 | 933.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 930.05 | 926.07 | 930.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 930.05 | 926.07 | 930.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 936.45 | 928.15 | 930.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 901.45 | 928.15 | 930.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 873.65 | 870.17 | 870.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 873.65 | 870.17 | 870.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 12:15:00 | 884.25 | 874.37 | 872.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 875.75 | 876.46 | 874.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 887.50 | 879.88 | 877.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 12:15:00 | 870.25 | 876.87 | 877.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 870.25 | 876.87 | 877.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 868.55 | 874.36 | 875.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 863.40 | 863.38 | 868.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:45:00 | 863.05 | 863.38 | 868.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 868.40 | 864.65 | 868.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 868.70 | 864.65 | 868.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 873.15 | 866.35 | 868.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 873.15 | 866.35 | 868.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 873.75 | 867.83 | 869.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 873.75 | 867.83 | 869.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 876.55 | 870.68 | 870.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 13:15:00 | 878.75 | 873.96 | 872.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 877.60 | 882.33 | 878.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 877.60 | 882.33 | 878.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 870.00 | 879.86 | 877.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 870.00 | 879.86 | 877.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 869.20 | 877.73 | 876.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 869.20 | 877.73 | 876.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 872.40 | 875.33 | 875.61 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 877.95 | 875.53 | 875.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 15:15:00 | 880.00 | 876.42 | 875.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 877.40 | 877.97 | 876.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 877.40 | 877.97 | 876.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 880.45 | 878.46 | 876.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 881.30 | 879.27 | 878.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 872.30 | 878.25 | 878.17 | SL hit (close<static) qty=1.00 sl=876.90 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 874.95 | 877.59 | 877.88 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 881.35 | 877.84 | 877.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 890.00 | 880.27 | 878.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 13:15:00 | 882.45 | 884.01 | 881.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 14:00:00 | 882.45 | 884.01 | 881.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 883.95 | 884.00 | 881.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 882.25 | 884.00 | 881.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 883.45 | 883.95 | 881.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:30:00 | 879.75 | 883.95 | 881.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 877.70 | 882.70 | 881.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 877.70 | 882.70 | 881.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 878.30 | 881.82 | 881.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 878.30 | 881.82 | 881.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 874.55 | 880.37 | 880.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 14:15:00 | 869.40 | 877.64 | 879.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 868.35 | 864.60 | 869.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 868.35 | 864.60 | 869.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 877.10 | 867.10 | 870.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 877.10 | 867.10 | 870.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 879.00 | 869.48 | 871.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 879.00 | 869.48 | 871.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 882.90 | 873.85 | 873.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 886.40 | 878.14 | 875.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 878.25 | 878.61 | 876.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 878.25 | 878.61 | 876.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 876.25 | 878.29 | 876.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 876.25 | 878.29 | 876.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 870.10 | 876.65 | 875.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 870.10 | 876.65 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 870.25 | 875.37 | 875.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 867.85 | 875.37 | 875.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 869.90 | 874.27 | 874.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 871.45 | 873.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 14:15:00 | 865.05 | 862.87 | 866.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 865.05 | 862.87 | 866.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 866.00 | 863.49 | 866.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 868.40 | 863.49 | 866.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 865.95 | 863.98 | 866.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 869.25 | 863.98 | 866.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 873.95 | 865.98 | 866.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 874.95 | 865.98 | 866.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 882.55 | 869.29 | 868.24 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 869.20 | 871.33 | 871.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 858.15 | 868.18 | 869.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 846.15 | 839.15 | 846.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 846.15 | 839.15 | 846.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 848.50 | 841.02 | 846.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 849.40 | 841.02 | 846.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 848.70 | 842.56 | 847.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 851.70 | 842.56 | 847.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 848.30 | 844.10 | 847.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 848.30 | 844.10 | 847.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 848.35 | 844.95 | 847.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 855.70 | 844.95 | 847.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 856.50 | 848.64 | 848.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 864.40 | 853.08 | 850.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 871.00 | 871.38 | 864.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 869.60 | 870.84 | 864.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 866.55 | 870.14 | 866.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 863.65 | 870.14 | 866.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 870.00 | 870.11 | 866.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 872.30 | 870.11 | 866.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 873.15 | 870.72 | 867.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 876.25 | 870.72 | 867.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 14:15:00 | 863.95 | 868.84 | 868.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 863.95 | 868.84 | 868.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 857.15 | 865.89 | 867.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 848.20 | 844.02 | 850.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 848.15 | 844.02 | 850.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 845.35 | 844.29 | 850.14 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 861.30 | 853.65 | 853.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 865.40 | 858.95 | 856.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 859.05 | 859.16 | 856.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:45:00 | 859.75 | 859.16 | 856.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 859.00 | 859.13 | 856.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:30:00 | 856.50 | 859.13 | 856.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 855.85 | 858.47 | 856.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 855.85 | 858.47 | 856.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 860.00 | 858.78 | 857.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 862.75 | 858.78 | 857.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 875.05 | 880.82 | 881.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 875.05 | 880.82 | 881.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 10:15:00 | 866.05 | 873.91 | 877.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 13:15:00 | 855.70 | 853.77 | 858.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:45:00 | 855.60 | 853.77 | 858.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 859.25 | 854.87 | 858.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 859.25 | 854.87 | 858.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 858.00 | 855.50 | 858.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 854.25 | 854.60 | 858.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 853.90 | 853.64 | 857.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 811.54 | 826.91 | 835.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 811.20 | 826.91 | 835.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 790.00 | 789.49 | 798.15 | SL hit (close>ema200) qty=0.50 sl=789.49 alert=retest2 |

### Cycle 47 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 757.70 | 749.59 | 748.76 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 744.20 | 749.50 | 750.20 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 754.25 | 749.52 | 749.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 755.70 | 751.93 | 750.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 765.00 | 765.61 | 762.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 764.55 | 765.61 | 762.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 759.95 | 764.48 | 762.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 759.95 | 764.48 | 762.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 765.10 | 764.60 | 762.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 768.50 | 765.48 | 763.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 767.50 | 765.98 | 764.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 758.30 | 765.35 | 765.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 758.30 | 765.35 | 765.82 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 775.05 | 766.77 | 765.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 781.95 | 771.82 | 769.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 783.50 | 784.66 | 779.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 786.80 | 784.44 | 780.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 786.55 | 784.86 | 780.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 789.80 | 790.05 | 787.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 784.15 | 786.77 | 786.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 784.15 | 786.77 | 786.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 770.60 | 782.92 | 785.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 780.15 | 775.16 | 779.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 780.15 | 775.16 | 779.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 783.60 | 776.85 | 780.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 774.30 | 776.85 | 780.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 778.10 | 777.45 | 779.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 778.50 | 777.45 | 779.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 778.45 | 777.51 | 779.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 783.85 | 778.78 | 779.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:45:00 | 784.45 | 778.78 | 779.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 789.35 | 780.89 | 780.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 789.35 | 780.89 | 780.67 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 778.25 | 780.64 | 780.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 772.80 | 779.07 | 780.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 774.90 | 772.88 | 775.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 774.90 | 772.88 | 775.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 782.60 | 774.83 | 776.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 755.95 | 774.83 | 776.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 718.15 | 740.85 | 754.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 730.70 | 728.20 | 737.36 | SL hit (close>ema200) qty=0.50 sl=728.20 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 712.20 | 701.70 | 700.38 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 694.30 | 699.26 | 699.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 12:15:00 | 685.70 | 693.44 | 696.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 694.00 | 691.66 | 694.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 673.00 | 691.66 | 694.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 702.25 | 677.82 | 677.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 702.25 | 677.82 | 677.20 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 674.70 | 684.63 | 685.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 662.40 | 678.50 | 682.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 640.50 | 633.93 | 643.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 640.50 | 633.93 | 643.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 636.65 | 633.34 | 636.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:00:00 | 636.65 | 633.34 | 636.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 632.80 | 633.24 | 636.23 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 666.00 | 641.40 | 639.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 681.75 | 670.25 | 662.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 662.80 | 673.85 | 669.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 668.35 | 673.85 | 669.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 669.75 | 672.82 | 669.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 669.05 | 671.78 | 668.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:45:00 | 668.00 | 671.32 | 669.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 672.10 | 671.48 | 669.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 688.45 | 671.56 | 669.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 677.80 | 686.29 | 686.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 677.80 | 686.29 | 686.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 674.20 | 683.87 | 685.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 12:15:00 | 681.90 | 680.96 | 683.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-21 12:30:00 | 680.45 | 680.96 | 683.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 680.95 | 680.96 | 683.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 681.65 | 680.96 | 683.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 678.55 | 679.69 | 682.07 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 689.00 | 683.33 | 683.29 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 682.00 | 683.84 | 684.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 672.30 | 680.82 | 682.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 675.00 | 673.65 | 677.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 677.20 | 673.65 | 677.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 679.80 | 674.88 | 677.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 679.80 | 674.88 | 677.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 675.60 | 675.03 | 677.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 670.85 | 675.50 | 677.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 637.31 | 646.26 | 654.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 15:15:00 | 645.00 | 644.26 | 650.21 | SL hit (close>ema200) qty=0.50 sl=644.26 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 649.90 | 646.44 | 646.20 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 645.55 | 646.61 | 646.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 645.20 | 646.12 | 646.40 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 12:00:00 | 914.15 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-05-27 14:00:00 | 913.30 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-05-29 11:45:00 | 913.25 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-29 14:15:00 | 912.30 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-30 09:30:00 | 916.50 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-03 11:45:00 | 916.80 | 2025-06-03 12:15:00 | 910.85 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-13 11:30:00 | 996.95 | 2025-06-17 11:15:00 | 994.25 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-13 13:15:00 | 995.85 | 2025-06-17 11:15:00 | 994.25 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-06-27 15:00:00 | 988.35 | 2025-06-30 13:15:00 | 960.30 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-07-21 14:00:00 | 896.10 | 2025-07-28 09:15:00 | 851.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 895.40 | 2025-07-28 09:15:00 | 850.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 15:15:00 | 895.50 | 2025-07-28 09:15:00 | 850.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:00:00 | 896.10 | 2025-07-31 09:15:00 | 806.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 895.40 | 2025-07-31 09:15:00 | 805.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-22 15:15:00 | 895.50 | 2025-07-31 09:15:00 | 805.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-01 09:15:00 | 800.25 | 2025-09-01 13:15:00 | 806.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-01 11:15:00 | 800.70 | 2025-09-01 13:15:00 | 806.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-03 11:15:00 | 796.20 | 2025-09-04 09:15:00 | 805.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-03 12:30:00 | 795.10 | 2025-09-04 09:15:00 | 805.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-04 12:15:00 | 796.10 | 2025-09-08 12:15:00 | 801.45 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-04 13:30:00 | 795.65 | 2025-09-08 12:15:00 | 801.45 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-15 12:45:00 | 874.40 | 2025-09-19 11:15:00 | 880.90 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-09-25 10:30:00 | 889.30 | 2025-09-26 09:15:00 | 869.10 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-09-29 11:15:00 | 873.20 | 2025-10-01 14:15:00 | 875.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-09-30 11:15:00 | 874.15 | 2025-10-03 10:15:00 | 874.80 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-09-30 12:00:00 | 872.65 | 2025-10-03 10:15:00 | 874.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-09-30 13:00:00 | 872.65 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-01 09:15:00 | 868.20 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-03 09:15:00 | 869.75 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-03 10:15:00 | 870.00 | 2025-10-03 11:15:00 | 875.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-09 15:00:00 | 923.05 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-10-10 13:45:00 | 924.00 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-10-10 15:00:00 | 924.95 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-13 10:00:00 | 922.45 | 2025-10-14 11:15:00 | 914.55 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-20 09:15:00 | 930.55 | 2025-10-23 15:15:00 | 927.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-10-27 09:15:00 | 901.45 | 2025-11-13 09:15:00 | 873.65 | STOP_HIT | 1.00 | 3.08% |
| BUY | retest2 | 2025-11-17 13:30:00 | 887.50 | 2025-11-18 12:15:00 | 870.25 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-11-28 11:45:00 | 881.30 | 2025-12-01 09:15:00 | 872.30 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-12-24 10:15:00 | 876.25 | 2025-12-26 14:15:00 | 863.95 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-01-02 09:15:00 | 862.75 | 2026-01-08 11:15:00 | 875.05 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2026-01-14 09:30:00 | 854.25 | 2026-01-20 13:15:00 | 811.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:30:00 | 853.90 | 2026-01-20 13:15:00 | 811.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 854.25 | 2026-01-23 10:15:00 | 790.00 | STOP_HIT | 0.50 | 7.52% |
| SELL | retest2 | 2026-01-14 11:30:00 | 853.90 | 2026-01-23 10:15:00 | 790.00 | STOP_HIT | 0.50 | 7.48% |
| BUY | retest2 | 2026-02-11 14:45:00 | 768.50 | 2026-02-13 12:15:00 | 758.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-12 10:15:00 | 767.50 | 2026-02-13 12:15:00 | 758.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-19 12:15:00 | 786.80 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-02-19 13:00:00 | 786.55 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-23 09:15:00 | 789.80 | 2026-02-23 14:15:00 | 784.15 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-02-25 09:15:00 | 774.30 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-25 10:30:00 | 778.10 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-25 11:00:00 | 778.50 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-25 12:45:00 | 778.45 | 2026-02-25 14:15:00 | 789.35 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-03-02 09:15:00 | 755.95 | 2026-03-04 11:15:00 | 718.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 755.95 | 2026-03-05 14:15:00 | 730.70 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-03-23 09:15:00 | 673.00 | 2026-03-25 09:15:00 | 702.25 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 668.35 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2026-04-13 10:45:00 | 669.75 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2026-04-13 12:15:00 | 669.05 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.31% |
| BUY | retest2 | 2026-04-13 12:45:00 | 668.00 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2026-04-15 09:15:00 | 688.45 | 2026-04-20 14:15:00 | 677.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-27 14:30:00 | 670.85 | 2026-04-30 10:15:00 | 637.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 14:30:00 | 670.85 | 2026-04-30 15:15:00 | 645.00 | STOP_HIT | 0.50 | 3.85% |
