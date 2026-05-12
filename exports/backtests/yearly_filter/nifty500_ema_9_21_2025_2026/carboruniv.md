# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1020.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 22 |
| ALERT3 | 127 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 83 |
| PARTIAL | 14 |
| TARGET_HIT | 4 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 54
- **Target hits / Stop hits / Partials:** 4 / 80 / 14
- **Avg / median % per leg:** 1.18% / -0.49%
- **Sum % (uncompounded):** 115.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 36 | 13 | 36.1% | 4 | 32 | 0 | 0.63% | 22.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 36 | 13 | 36.1% | 4 | 32 | 0 | 0.63% | 22.8% |
| SELL (all) | 62 | 31 | 50.0% | 0 | 48 | 14 | 1.50% | 92.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.74% | 11.5% |
| SELL @ 3rd Alert (retest2) | 60 | 29 | 48.3% | 0 | 47 | 13 | 1.36% | 81.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.74% | 11.5% |
| retest2 (combined) | 96 | 42 | 43.8% | 4 | 79 | 13 | 1.09% | 104.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 990.80 | 979.12 | 977.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 996.00 | 984.72 | 980.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 965.10 | 979.09 | 980.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 11:15:00 | 960.50 | 975.37 | 979.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 978.10 | 969.79 | 974.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 988.70 | 973.57 | 975.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 988.70 | 973.57 | 975.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 981.10 | 975.08 | 975.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 991.00 | 975.08 | 975.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 987.30 | 977.52 | 977.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 989.40 | 981.33 | 979.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 982.00 | 982.41 | 980.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:30:00 | 981.20 | 982.41 | 980.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 991.50 | 984.31 | 981.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 987.30 | 984.31 | 981.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 995.90 | 995.56 | 990.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 997.00 | 995.56 | 990.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 995.20 | 995.49 | 990.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 992.60 | 995.49 | 990.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 994.10 | 1000.09 | 996.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 994.10 | 1000.09 | 996.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 998.50 | 999.77 | 996.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1005.40 | 999.77 | 996.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1001.20 | 1003.46 | 999.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 1003.20 | 1000.31 | 999.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 1002.00 | 1000.49 | 999.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1000.00 | 1000.71 | 1000.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1000.10 | 1000.71 | 1000.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 999.50 | 1000.47 | 1000.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 994.50 | 998.56 | 999.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 1002.30 | 999.00 | 999.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 999.50 | 999.10 | 999.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1000.00 | 999.10 | 999.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 999.00 | 999.08 | 999.27 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1001.20 | 999.50 | 999.45 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 997.70 | 999.24 | 999.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 993.10 | 997.99 | 998.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 1000.20 | 997.85 | 998.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 995.50 | 997.38 | 998.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 999.30 | 997.38 | 998.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 994.20 | 995.18 | 996.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:15:00 | 988.70 | 995.18 | 996.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 991.30 | 988.97 | 992.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 989.00 | 988.16 | 989.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 990.60 | 983.40 | 985.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 986.40 | 984.00 | 985.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 981.60 | 983.39 | 985.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.26 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.73 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.55 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.07 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 932.52 | 943.16 | 948.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 11:15:00 | 943.40 | 938.63 | 943.51 | SL hit (close>ema200) qty=0.50 sl=938.63 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 950.55 | 943.04 | 942.72 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 937.15 | 942.36 | 942.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 930.00 | 938.85 | 940.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 13:15:00 | 936.25 | 935.42 | 938.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 936.25 | 935.42 | 938.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 938.05 | 935.94 | 938.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 936.25 | 935.94 | 938.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 931.35 | 934.85 | 937.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 928.60 | 933.30 | 936.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 924.95 | 930.08 | 934.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 939.35 | 929.63 | 931.07 | SL hit (close>static) qty=1.00 sl=938.35 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 942.10 | 932.01 | 930.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 954.35 | 936.48 | 933.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 967.05 | 968.35 | 957.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 967.05 | 968.35 | 957.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 960.70 | 964.92 | 959.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 960.85 | 964.92 | 959.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 964.50 | 964.84 | 960.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 976.70 | 964.97 | 960.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 973.00 | 965.14 | 962.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 968.40 | 963.77 | 962.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:15:00 | 969.10 | 964.57 | 963.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 964.50 | 964.56 | 963.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 964.50 | 964.56 | 963.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 974.40 | 966.53 | 964.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 982.35 | 966.53 | 964.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 980.45 | 988.02 | 989.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 985.00 | 984.91 | 987.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 995.00 | 984.91 | 987.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 987.00 | 985.33 | 987.27 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 993.00 | 987.76 | 987.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 999.20 | 990.05 | 988.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 992.15 | 992.41 | 990.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:15:00 | 991.25 | 992.41 | 990.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 999.80 | 993.89 | 991.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 992.00 | 993.89 | 991.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 997.40 | 995.06 | 992.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 996.60 | 995.06 | 992.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 991.75 | 995.13 | 993.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 991.75 | 995.13 | 993.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 983.00 | 992.70 | 992.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 979.00 | 986.80 | 989.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 983.80 | 981.95 | 986.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 983.80 | 981.95 | 986.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 982.85 | 982.13 | 986.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 976.90 | 981.51 | 985.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 978.00 | 980.81 | 984.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 995.80 | 984.44 | 985.83 | SL hit (close>static) qty=1.00 sl=986.45 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 993.05 | 987.13 | 986.87 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 989.85 | 993.49 | 993.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 985.60 | 991.33 | 992.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 995.10 | 991.99 | 992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 988.35 | 991.26 | 992.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 986.40 | 989.86 | 991.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 983.75 | 988.64 | 990.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 937.08 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 934.56 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 940.00 | 933.66 | 942.91 | SL hit (close>ema200) qty=0.50 sl=933.66 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 947.55 | 944.81 | 944.53 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 930.40 | 943.94 | 944.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 926.25 | 936.47 | 939.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 920.85 | 920.53 | 927.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:00:00 | 910.55 | 918.54 | 925.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 865.02 | 883.60 | 898.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 851.60 | 846.99 | 860.14 | SL hit (close>ema200) qty=0.50 sl=846.99 alert=retest1 |

### Cycle 17 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 869.90 | 853.22 | 852.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 880.00 | 858.57 | 854.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 11:15:00 | 871.20 | 871.49 | 864.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:30:00 | 871.95 | 871.49 | 864.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 874.95 | 872.18 | 865.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 875.50 | 871.87 | 866.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 875.55 | 871.89 | 867.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-20 11:15:00 | 963.05 | 905.72 | 885.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 913.50 | 927.76 | 928.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 904.10 | 914.94 | 921.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 905.00 | 904.07 | 910.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 905.75 | 904.07 | 910.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 911.20 | 905.50 | 910.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 911.20 | 905.50 | 910.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 919.15 | 908.23 | 911.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 919.15 | 908.23 | 911.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 918.60 | 910.30 | 912.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 915.00 | 910.30 | 912.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 922.00 | 914.85 | 913.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 948.70 | 949.92 | 940.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 948.70 | 949.92 | 940.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 945.00 | 948.94 | 941.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 947.50 | 948.94 | 941.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 945.80 | 948.31 | 941.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 957.45 | 948.48 | 944.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 953.05 | 949.48 | 946.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 956.10 | 949.48 | 946.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 962.30 | 954.86 | 949.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 970.60 | 966.15 | 959.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 959.50 | 966.15 | 959.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 988.35 | 992.71 | 987.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 990.00 | 992.71 | 987.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 985.80 | 991.33 | 987.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 10:15:00 | 973.95 | 980.04 | 982.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 980.00 | 978.91 | 981.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 980.00 | 978.91 | 981.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 21 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 999.90 | 983.11 | 982.96 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 982.10 | 989.21 | 989.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 977.80 | 984.36 | 986.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 15:15:00 | 957.00 | 955.50 | 965.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 962.40 | 955.50 | 965.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 955.75 | 955.55 | 964.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 962.45 | 955.55 | 964.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 970.00 | 959.39 | 965.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 968.50 | 959.39 | 965.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 971.00 | 961.71 | 965.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 971.00 | 961.71 | 965.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 974.40 | 968.37 | 967.96 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 963.60 | 967.20 | 967.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 960.40 | 965.84 | 966.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 924.90 | 919.75 | 928.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 925.75 | 920.99 | 926.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 925.75 | 920.99 | 926.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 921.65 | 921.12 | 925.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 920.60 | 921.12 | 925.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 918.90 | 920.37 | 924.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 919.30 | 916.95 | 921.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 918.65 | 917.94 | 920.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 920.95 | 918.54 | 920.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 920.95 | 918.54 | 920.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | SL hit (close>static) qty=1.00 sl=926.35 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 930.25 | 922.63 | 922.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 938.60 | 927.19 | 924.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 931.10 | 931.30 | 927.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 931.10 | 931.30 | 927.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 928.00 | 930.64 | 927.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 927.25 | 930.64 | 927.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 921.20 | 928.75 | 926.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 921.20 | 928.75 | 926.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 921.20 | 927.24 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 922.00 | 927.24 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 919.90 | 925.78 | 925.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 914.90 | 921.59 | 923.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 929.85 | 922.17 | 923.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 919.25 | 921.59 | 922.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 918.25 | 921.59 | 922.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 918.70 | 920.24 | 921.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 915.45 | 914.01 | 915.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 915.80 | 914.01 | 915.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 921.50 | 915.51 | 916.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 920.25 | 915.51 | 916.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 916.40 | 915.68 | 916.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 939.30 | 922.39 | 919.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 913.40 | 921.61 | 922.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 910.00 | 916.24 | 918.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 915.65 | 914.45 | 916.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 915.65 | 914.45 | 916.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 916.55 | 914.87 | 916.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 916.50 | 914.87 | 916.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 917.25 | 915.35 | 916.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 917.40 | 915.35 | 916.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 912.95 | 914.87 | 916.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 909.00 | 912.38 | 914.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 908.00 | 903.15 | 906.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 914.90 | 907.37 | 905.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 909.55 | 909.68 | 907.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 909.55 | 909.68 | 907.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 909.50 | 909.49 | 907.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 903.35 | 909.49 | 907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 910.25 | 909.64 | 907.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 907.20 | 909.64 | 907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 910.80 | 909.87 | 908.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:00:00 | 914.65 | 910.83 | 908.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 914.70 | 912.17 | 909.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 897.65 | 910.79 | 913.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 907.95 | 905.94 | 909.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:45:00 | 908.50 | 905.94 | 909.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 907.95 | 906.34 | 909.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 908.35 | 906.34 | 909.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 910.40 | 907.15 | 909.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 910.40 | 907.15 | 909.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 909.90 | 907.70 | 909.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 915.40 | 907.70 | 909.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 913.95 | 908.95 | 910.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 916.55 | 908.95 | 910.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 922.10 | 911.58 | 911.15 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 907.85 | 910.84 | 910.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 901.50 | 908.97 | 910.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 895.00 | 890.13 | 895.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 904.50 | 893.01 | 896.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 904.50 | 893.01 | 896.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 905.50 | 895.51 | 897.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 905.50 | 895.51 | 897.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 916.10 | 899.62 | 898.81 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 893.45 | 903.09 | 904.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 15:15:00 | 890.10 | 897.81 | 901.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 885.00 | 892.42 | 894.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 888.00 | 889.74 | 892.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 885.05 | 886.75 | 889.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 843.60 | 856.72 | 867.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.75 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.80 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 830.90 | 826.71 | 834.01 | SL hit (close>ema200) qty=0.50 sl=826.71 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 835.15 | 833.19 | 833.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 874.50 | 841.45 | 836.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 874.00 | 875.55 | 859.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 873.85 | 875.55 | 859.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 868.70 | 871.60 | 861.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 870.60 | 871.40 | 862.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 870.00 | 870.77 | 863.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 869.90 | 870.02 | 863.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 855.00 | 867.00 | 863.26 | SL hit (close<static) qty=1.00 sl=861.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 851.50 | 859.93 | 860.63 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 871.00 | 861.30 | 860.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 874.90 | 865.92 | 863.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 872.85 | 875.09 | 869.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 872.85 | 875.09 | 869.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 871.60 | 874.39 | 869.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 870.95 | 874.39 | 869.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 885.10 | 876.53 | 871.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:00:00 | 888.15 | 881.79 | 875.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 891.20 | 883.10 | 876.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 888.45 | 885.70 | 879.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 894.25 | 886.06 | 880.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 880.10 | 885.55 | 882.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 880.95 | 885.55 | 882.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 878.25 | 884.09 | 881.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:15:00 | 876.15 | 884.09 | 881.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 878.90 | 883.05 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 877.20 | 883.05 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 887.00 | 883.84 | 882.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 878.40 | 883.84 | 882.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 875.10 | 882.09 | 881.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 873.00 | 882.09 | 881.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 883.40 | 882.35 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 882.90 | 882.35 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 873.20 | 879.35 | 880.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 850.55 | 848.23 | 857.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 850.55 | 848.23 | 857.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 874.90 | 853.86 | 857.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 874.90 | 853.86 | 857.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 877.00 | 858.49 | 859.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 863.55 | 858.49 | 859.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 873.90 | 865.78 | 862.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 869.80 | 871.42 | 867.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 869.80 | 871.42 | 867.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 859.10 | 869.04 | 867.48 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 862.20 | 866.20 | 866.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 858.10 | 862.63 | 864.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 849.55 | 846.75 | 852.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 849.55 | 846.75 | 852.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 852.00 | 847.80 | 852.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 848.00 | 849.18 | 852.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 859.80 | 852.22 | 852.91 | SL hit (close>static) qty=1.00 sl=856.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 853.45 | 850.83 | 850.54 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 846.00 | 849.84 | 850.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 842.40 | 847.92 | 849.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 850.45 | 848.42 | 849.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 853.85 | 849.51 | 849.73 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 851.75 | 849.96 | 849.91 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 847.75 | 849.52 | 849.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 837.40 | 847.06 | 848.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 845.60 | 834.67 | 837.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 853.00 | 838.33 | 839.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 853.00 | 838.33 | 839.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 852.10 | 841.09 | 840.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 853.20 | 843.51 | 841.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:15:00 | 845.10 | 847.80 | 845.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 845.00 | 847.24 | 845.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 848.95 | 849.58 | 846.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 850.30 | 858.25 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 837.35 | 846.52 | 851.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 807.20 | 804.36 | 812.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 805.85 | 804.36 | 812.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 808.25 | 803.81 | 809.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 814.05 | 803.81 | 809.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 809.50 | 804.95 | 809.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 804.80 | 808.19 | 809.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 820.75 | 813.54 | 811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 803.35 | 814.25 | 812.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 805.70 | 812.54 | 811.77 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 803.10 | 809.70 | 810.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 795.75 | 806.50 | 808.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 812.35 | 798.15 | 802.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 805.80 | 799.68 | 803.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 810.60 | 799.68 | 803.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 806.10 | 802.87 | 804.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:15:00 | 802.50 | 802.87 | 804.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 801.40 | 803.67 | 803.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 830.15 | 820.23 | 816.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 821.00 | 827.88 | 822.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 816.65 | 825.63 | 822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 816.10 | 825.63 | 822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 815.65 | 821.85 | 821.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 815.65 | 821.85 | 821.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 804.30 | 818.34 | 819.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 799.00 | 812.31 | 816.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 798.15 | 794.62 | 802.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:00:00 | 798.15 | 794.62 | 802.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 777.15 | 765.80 | 777.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 782.00 | 765.80 | 777.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 787.10 | 770.06 | 778.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 787.10 | 770.06 | 778.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 799.00 | 775.85 | 780.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 799.00 | 775.85 | 780.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 805.75 | 785.78 | 784.05 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 765.15 | 787.38 | 790.18 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 821.95 | 788.56 | 786.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 827.25 | 796.30 | 790.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 823.00 | 823.84 | 814.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 820.25 | 823.84 | 814.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 821.10 | 830.26 | 823.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 823.65 | 830.26 | 823.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 831.60 | 830.53 | 824.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 837.85 | 830.53 | 824.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 833.50 | 834.10 | 829.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 829.00 | 826.42 | 826.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 831.20 | 828.49 | 827.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 832.25 | 841.09 | 837.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 841.00 | 841.08 | 837.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 857.00 | 843.27 | 838.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 850.00 | 848.44 | 844.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 849.35 | 848.01 | 844.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 831.55 | 842.09 | 843.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 842.95 | 842.00 | 843.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 842.95 | 842.00 | 843.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 841.15 | 841.89 | 843.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 837.20 | 841.27 | 842.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:30:00 | 837.30 | 838.72 | 841.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:45:00 | 838.00 | 835.92 | 837.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.34 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.43 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 796.10 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 813.00 | 806.62 | 815.78 | SL hit (close>ema200) qty=0.50 sl=806.62 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 813.00 | 808.01 | 807.75 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 805.40 | 807.49 | 807.54 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 808.15 | 807.62 | 807.60 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 805.45 | 807.17 | 807.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 804.25 | 806.60 | 807.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 797.40 | 794.55 | 799.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 799.95 | 795.63 | 799.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 798.90 | 795.63 | 799.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 800.00 | 796.50 | 799.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 801.30 | 796.50 | 799.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 806.40 | 798.48 | 800.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 807.75 | 798.48 | 800.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 804.00 | 799.58 | 800.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 800.85 | 799.58 | 800.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 798.55 | 800.71 | 800.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 796.95 | 799.95 | 800.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 790.00 | 786.28 | 791.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 786.85 | 786.39 | 790.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 773.45 | 786.63 | 790.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 754.05 | 759.85 | 760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 748.45 | 756.30 | 758.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 749.80 | 754.19 | 756.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 747.80 | 757.21 | 757.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 12:15:00 | 748.65 | 754.33 | 755.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 766.50 | 757.74 | 756.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 781.50 | 759.38 | 757.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 768.40 | 770.82 | 768.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 766.15 | 769.65 | 768.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 11:15:00 | 771.15 | 768.50 | 768.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 777.15 | 771.68 | 769.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 13:15:00 | 832.05 | 833.68 | 820.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 14:00:00 | 832.05 | 833.68 | 820.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 833.75 | 831.69 | 822.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 835.35 | 831.69 | 822.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:15:00 | 838.55 | 832.30 | 823.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 14:15:00 | 918.89 | 873.26 | 864.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 929.15 | 939.69 | 940.19 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 946.25 | 939.33 | 939.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 952.70 | 945.64 | 942.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 952.35 | 968.48 | 962.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 946.25 | 964.03 | 960.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 945.95 | 964.03 | 960.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 948.90 | 958.20 | 958.56 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 972.00 | 959.10 | 958.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 987.95 | 973.94 | 969.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 13:15:00 | 1005.40 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1001.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-05-23 12:00:00 | 1003.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-23 12:45:00 | 1002.00 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-12 11:15:00 | 939.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-12 11:15:00 | 941.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-12 11:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-12 11:15:00 | 941.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-12 13:15:00 | 932.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-06-19 10:30:00 | 928.60 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-19 12:45:00 | 924.95 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-20 15:15:00 | 923.90 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-06-23 09:30:00 | 926.35 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-27 10:15:00 | 976.70 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-06-30 09:15:00 | 973.00 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-30 12:30:00 | 968.40 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-06-30 14:15:00 | 969.10 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-07-01 09:15:00 | 982.35 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-14 12:15:00 | 976.90 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-07-14 13:00:00 | 978.00 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-28 13:15:00 | 937.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-28 13:15:00 | 934.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-07 10:15:00 | 865.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-11 11:15:00 | 851.60 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2025-08-12 11:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-12 12:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-13 13:30:00 | 850.90 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-14 10:30:00 | 852.85 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-19 14:30:00 | 875.50 | 2025-08-20 11:15:00 | 963.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-20 09:15:00 | 875.55 | 2025-08-20 11:15:00 | 963.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 12:15:00 | 915.00 | 2025-09-01 10:15:00 | 916.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-05 09:45:00 | 957.45 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-09-05 13:30:00 | 953.05 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2025-09-05 14:15:00 | 956.10 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2025-09-08 09:30:00 | 962.30 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2025-10-01 09:15:00 | 920.60 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-01 09:45:00 | 918.90 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-01 14:45:00 | 919.30 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-03 10:00:00 | 918.65 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-08 11:15:00 | 918.25 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-08 12:45:00 | 918.70 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-10 09:45:00 | 915.45 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-10 10:15:00 | 915.80 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-17 11:30:00 | 909.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-10-21 14:15:00 | 908.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-29 12:00:00 | 914.65 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-29 13:30:00 | 914.70 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-19 10:15:00 | 843.60 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-19 12:15:00 | 840.75 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-19 12:15:00 | 840.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.12% |
| BUY | retest2 | 2025-11-27 13:00:00 | 870.60 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-27 14:15:00 | 870.00 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-27 15:15:00 | 869.90 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-12-03 10:00:00 | 888.15 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-03 10:30:00 | 891.20 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-03 14:00:00 | 888.45 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-03 15:15:00 | 894.25 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-12-10 09:15:00 | 863.55 | 2025-12-11 14:15:00 | 869.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-19 09:30:00 | 848.00 | 2025-12-19 13:15:00 | 859.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-22 10:00:00 | 848.00 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-23 09:15:00 | 844.80 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-23 14:45:00 | 847.70 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-01-01 12:30:00 | 848.95 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-06 09:45:00 | 850.30 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-14 14:30:00 | 804.80 | 2026-01-16 09:15:00 | 816.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-21 13:15:00 | 802.50 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-22 09:30:00 | 801.40 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-02-12 12:15:00 | 837.85 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-13 12:00:00 | 833.50 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-02-20 09:45:00 | 857.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-23 09:15:00 | 850.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-23 10:45:00 | 849.35 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 09:15:00 | 795.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 09:15:00 | 795.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 09:15:00 | 796.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-10 12:15:00 | 800.85 | 2026-03-11 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-13 09:15:00 | 773.45 | 2026-03-18 13:15:00 | 770.20 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 10:45:00 | 749.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-03-23 09:15:00 | 747.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-23 12:15:00 | 748.65 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-03-25 09:15:00 | 781.50 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-27 13:15:00 | 768.40 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-03-27 14:15:00 | 766.15 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-04-07 10:15:00 | 835.35 | 2026-04-10 14:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:15:00 | 838.55 | 2026-04-17 10:15:00 | 922.40 | TARGET_HIT | 1.00 | 10.00% |
