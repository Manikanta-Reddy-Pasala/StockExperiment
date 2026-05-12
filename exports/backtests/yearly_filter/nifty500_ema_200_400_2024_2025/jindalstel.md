# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1248.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 62 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 2 |
| TARGET_HIT | 15 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 32
- **Target hits / Stop hits / Partials:** 14 / 33 / 2
- **Avg / median % per leg:** 1.78% / -1.29%
- **Sum % (uncompounded):** 87.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.06% | 121.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.06% | 121.1% |
| SELL (all) | 29 | 3 | 10.3% | 0 | 27 | 2 | -1.17% | -33.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 3 | 10.3% | 0 | 27 | 2 | -1.17% | -33.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 17 | 34.7% | 14 | 33 | 2 | 1.78% | 87.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 938.95 | 983.04 | 983.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 928.00 | 980.53 | 981.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 14:15:00 | 958.20 | 958.05 | 968.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 15:00:00 | 958.20 | 958.05 | 968.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 965.80 | 957.98 | 967.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 965.00 | 957.98 | 967.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 963.55 | 958.03 | 967.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-23 15:15:00 | 957.80 | 958.30 | 967.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 12:15:00 | 974.50 | 958.65 | 967.67 | SL hit (close>static) qty=1.00 sl=969.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 1041.60 | 970.59 | 970.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 1050.95 | 972.07 | 971.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 999.30 | 1007.45 | 992.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 10:15:00 | 999.30 | 1007.45 | 992.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 999.30 | 1007.45 | 992.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 999.30 | 1007.45 | 992.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 997.25 | 1007.34 | 992.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:30:00 | 998.40 | 1007.34 | 992.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 985.00 | 1006.83 | 993.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:45:00 | 1003.25 | 1005.44 | 992.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 1000.40 | 1005.39 | 992.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 14:45:00 | 1004.95 | 1005.26 | 993.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 1014.25 | 1005.20 | 993.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 993.95 | 1004.98 | 993.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:00:00 | 993.95 | 1004.98 | 993.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 990.20 | 1004.84 | 993.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 990.20 | 1004.84 | 993.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 992.00 | 1004.71 | 993.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 999.65 | 1004.71 | 993.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:30:00 | 995.65 | 1004.68 | 993.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 988.05 | 1004.37 | 993.69 | SL hit (close<static) qty=1.00 sl=988.85 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 911.00 | 984.98 | 985.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 898.00 | 976.02 | 980.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 960.75 | 952.47 | 965.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 960.75 | 952.47 | 965.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 960.75 | 952.47 | 965.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:45:00 | 957.65 | 952.47 | 965.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 926.75 | 913.07 | 935.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 931.55 | 913.07 | 935.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 928.15 | 914.41 | 934.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:00:00 | 928.15 | 914.41 | 934.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 935.25 | 914.77 | 934.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 933.00 | 914.77 | 934.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 934.40 | 914.96 | 934.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 932.65 | 914.96 | 934.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 926.50 | 915.08 | 934.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 913.10 | 942.93 | 945.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 12:30:00 | 923.30 | 942.02 | 944.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 923.45 | 940.83 | 943.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:30:00 | 921.85 | 940.60 | 943.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 944.50 | 939.96 | 943.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 944.50 | 939.96 | 943.46 | SL hit (close>static) qty=1.00 sl=939.30 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 923.00 | 888.66 | 888.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 13:15:00 | 796.35 | 889.17 | 889.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 09:15:00 | 770.25 | 880.06 | 884.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-17 10:15:00 | 887.20 | 866.64 | 876.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 10:15:00 | 887.20 | 866.64 | 876.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 887.20 | 866.64 | 876.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 887.20 | 866.64 | 876.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 887.00 | 866.84 | 876.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-17 14:30:00 | 882.90 | 867.36 | 877.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 10:15:00 | 896.70 | 868.00 | 877.21 | SL hit (close>static) qty=1.00 sl=895.95 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 902.30 | 884.38 | 884.30 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 881.65 | 884.23 | 884.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 14:15:00 | 877.75 | 884.14 | 884.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 881.40 | 880.43 | 882.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 881.40 | 880.43 | 882.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 881.40 | 880.43 | 882.27 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 927.75 | 884.09 | 884.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 948.70 | 885.54 | 884.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 937.20 | 942.23 | 923.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 937.20 | 942.23 | 923.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 933.00 | 942.08 | 923.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:15:00 | 934.95 | 927.54 | 919.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 936.30 | 927.55 | 920.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 11:00:00 | 934.70 | 939.65 | 929.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 937.35 | 939.48 | 929.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 928.15 | 939.31 | 930.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 928.15 | 939.31 | 930.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 928.95 | 939.21 | 930.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:45:00 | 928.05 | 939.21 | 930.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 932.20 | 939.14 | 930.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 935.90 | 939.14 | 930.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 939.50 | 939.11 | 930.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:30:00 | 935.15 | 939.26 | 930.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 938.10 | 939.19 | 930.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 933.00 | 939.08 | 930.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 931.25 | 939.08 | 930.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 944.70 | 958.23 | 944.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 944.70 | 958.23 | 944.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 946.95 | 958.12 | 944.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 967.00 | 958.12 | 944.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:45:00 | 948.30 | 980.31 | 965.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 949.20 | 980.31 | 965.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-03 14:15:00 | 1028.45 | 980.31 | 967.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1012.70 | 1031.40 | 1031.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1007.80 | 1030.95 | 1031.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1021.30 | 1016.55 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 1022.40 | 1016.55 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1022.10 | 1016.61 | 1022.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 1023.00 | 1016.61 | 1022.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1020.70 | 1016.65 | 1022.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1059.90 | 1016.65 | 1022.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1052.90 | 1017.01 | 1022.97 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 1077.80 | 1028.55 | 1028.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 14:15:00 | 1082.30 | 1029.09 | 1028.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1034.40 | 1036.05 | 1032.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1034.40 | 1036.05 | 1032.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1031.60 | 1036.00 | 1032.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1031.60 | 1036.00 | 1032.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1032.00 | 1035.97 | 1032.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 1035.80 | 1031.33 | 1030.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 11:00:00 | 1039.30 | 1035.13 | 1032.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 14:30:00 | 1037.70 | 1035.25 | 1032.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-29 09:15:00 | 1139.38 | 1048.19 | 1039.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-23 15:15:00 | 957.80 | 2024-08-26 12:15:00 | 974.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-08-29 10:30:00 | 959.05 | 2024-08-30 10:15:00 | 973.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-29 11:45:00 | 959.05 | 2024-08-30 10:15:00 | 973.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-29 12:30:00 | 958.85 | 2024-08-30 10:15:00 | 973.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-09-03 10:30:00 | 958.65 | 2024-09-10 12:15:00 | 970.55 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-09-06 12:45:00 | 958.20 | 2024-09-10 12:15:00 | 970.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-09-06 13:45:00 | 957.40 | 2024-09-10 12:15:00 | 970.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-10 10:45:00 | 956.25 | 2024-09-10 12:15:00 | 970.55 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-10-09 10:45:00 | 1003.25 | 2024-10-14 12:15:00 | 988.05 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-10-09 11:45:00 | 1000.40 | 2024-10-14 12:15:00 | 988.05 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-10-09 14:45:00 | 1004.95 | 2024-10-16 09:15:00 | 963.00 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2024-10-10 09:15:00 | 1014.25 | 2024-10-16 09:15:00 | 963.00 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest2 | 2024-10-11 09:15:00 | 999.65 | 2024-10-16 09:15:00 | 963.00 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2024-10-14 10:30:00 | 995.65 | 2024-10-16 09:15:00 | 963.00 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2024-12-19 09:15:00 | 913.10 | 2024-12-23 10:15:00 | 944.50 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-12-19 12:30:00 | 923.30 | 2024-12-23 10:15:00 | 944.50 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-12-20 13:00:00 | 923.45 | 2024-12-23 10:15:00 | 944.50 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-12-20 13:30:00 | 921.85 | 2024-12-23 10:15:00 | 944.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2024-12-23 12:30:00 | 935.55 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-12-23 13:45:00 | 934.65 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-12-24 09:15:00 | 933.00 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-24 10:45:00 | 936.20 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-12-24 14:30:00 | 939.50 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-26 09:30:00 | 940.00 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-26 14:30:00 | 939.10 | 2024-12-27 09:15:00 | 947.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-12-27 10:30:00 | 939.40 | 2025-01-03 11:15:00 | 952.95 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-01-08 12:00:00 | 934.50 | 2025-01-13 13:15:00 | 887.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 15:15:00 | 936.00 | 2025-01-13 13:15:00 | 889.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:00:00 | 934.50 | 2025-01-20 12:15:00 | 935.00 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2025-01-08 15:15:00 | 936.00 | 2025-01-20 12:15:00 | 935.00 | STOP_HIT | 0.50 | 0.11% |
| SELL | retest2 | 2025-04-17 14:30:00 | 882.90 | 2025-04-21 10:15:00 | 896.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-04-25 11:15:00 | 880.50 | 2025-04-28 09:15:00 | 905.85 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-05-02 09:15:00 | 872.05 | 2025-05-02 13:15:00 | 901.70 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-05-02 10:45:00 | 880.95 | 2025-05-02 13:15:00 | 901.70 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-05-02 15:15:00 | 892.00 | 2025-05-05 11:15:00 | 907.75 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-06-24 11:15:00 | 934.95 | 2025-09-03 14:15:00 | 1028.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 12:45:00 | 936.30 | 2025-09-03 14:15:00 | 1029.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 11:00:00 | 934.70 | 2025-09-03 14:15:00 | 1028.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-11 13:45:00 | 937.35 | 2025-09-03 14:15:00 | 1029.49 | TARGET_HIT | 1.00 | 9.83% |
| BUY | retest2 | 2025-07-14 15:15:00 | 935.90 | 2025-09-03 14:15:00 | 1028.66 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2025-07-15 09:45:00 | 939.50 | 2025-09-04 09:15:00 | 1031.09 | TARGET_HIT | 1.00 | 9.75% |
| BUY | retest2 | 2025-07-16 14:30:00 | 935.15 | 2025-09-04 09:15:00 | 1033.45 | TARGET_HIT | 1.00 | 10.51% |
| BUY | retest2 | 2025-07-17 09:15:00 | 938.10 | 2025-09-04 09:15:00 | 1031.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 967.00 | 2025-09-08 09:15:00 | 1043.13 | TARGET_HIT | 1.00 | 7.87% |
| BUY | retest2 | 2025-08-29 14:45:00 | 948.30 | 2025-09-08 09:15:00 | 1044.12 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2025-08-29 15:15:00 | 949.20 | 2025-09-23 13:15:00 | 1063.70 | TARGET_HIT | 1.00 | 12.06% |
| BUY | retest2 | 2026-01-14 10:30:00 | 1035.80 | 2026-01-29 09:15:00 | 1139.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 11:00:00 | 1039.30 | 2026-01-29 09:15:00 | 1143.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 14:30:00 | 1037.70 | 2026-01-29 09:15:00 | 1141.47 | TARGET_HIT | 1.00 | 10.00% |
