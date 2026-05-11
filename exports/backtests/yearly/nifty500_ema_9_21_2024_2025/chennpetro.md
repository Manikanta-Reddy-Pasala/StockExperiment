# Chennai Petroleum Corporation Ltd. (CHENNPETRO)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1079.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 122 |
| ALERT1 | 93 |
| ALERT2 | 93 |
| ALERT2_SKIP | 49 |
| ALERT3 | 226 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 125 |
| PARTIAL | 20 |
| TARGET_HIT | 12 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 146 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 54 / 92
- **Target hits / Stop hits / Partials:** 12 / 114 / 20
- **Avg / median % per leg:** 0.62% / -0.51%
- **Sum % (uncompounded):** 90.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 18 | 31.6% | 8 | 49 | 0 | 0.55% | 31.4% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -1.36% | -4.1% |
| BUY @ 3rd Alert (retest2) | 54 | 17 | 31.5% | 8 | 46 | 0 | 0.66% | 35.5% |
| SELL (all) | 89 | 36 | 40.4% | 4 | 65 | 20 | 0.67% | 59.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.15% | -1.1% |
| SELL @ 3rd Alert (retest2) | 88 | 36 | 40.9% | 4 | 64 | 20 | 0.69% | 60.4% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -1.31% | -5.2% |
| retest2 (combined) | 142 | 53 | 37.3% | 12 | 110 | 20 | 0.68% | 95.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 907.05 | 880.55 | 877.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 915.15 | 907.64 | 899.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 905.60 | 907.88 | 901.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 905.60 | 907.88 | 901.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 890.55 | 904.41 | 900.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:00:00 | 890.55 | 904.41 | 900.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 897.55 | 903.04 | 900.20 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 09:15:00 | 883.05 | 898.30 | 898.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 10:15:00 | 882.00 | 895.04 | 897.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-18 09:15:00 | 889.45 | 884.95 | 889.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 889.45 | 884.95 | 889.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 889.45 | 884.95 | 889.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 896.00 | 887.71 | 890.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 898.00 | 889.77 | 891.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 887.20 | 889.77 | 891.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 10:00:00 | 894.55 | 890.73 | 891.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 11:15:00 | 896.75 | 892.68 | 892.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 11:15:00 | 896.75 | 892.68 | 892.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 904.10 | 894.96 | 893.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 892.00 | 894.50 | 893.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 892.00 | 894.50 | 893.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 892.00 | 894.50 | 893.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:30:00 | 889.05 | 894.50 | 893.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 889.00 | 893.40 | 893.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 893.50 | 893.40 | 893.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 893.95 | 894.05 | 893.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 891.60 | 894.05 | 893.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 894.50 | 894.14 | 893.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 893.20 | 894.14 | 893.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 894.00 | 894.11 | 893.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 892.75 | 894.11 | 893.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 893.90 | 894.07 | 893.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 893.90 | 894.07 | 893.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 892.65 | 893.79 | 893.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 892.30 | 893.79 | 893.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 891.10 | 893.25 | 893.36 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 912.00 | 897.00 | 895.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 931.45 | 908.55 | 901.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 960.35 | 964.53 | 944.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 09:45:00 | 953.85 | 964.53 | 944.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 960.60 | 965.45 | 958.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 956.50 | 965.45 | 958.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 958.00 | 963.32 | 958.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 958.00 | 963.32 | 958.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 960.80 | 962.81 | 958.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 966.30 | 962.81 | 958.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 960.90 | 962.43 | 958.85 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 14:15:00 | 949.90 | 957.85 | 957.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 932.10 | 951.16 | 954.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 922.65 | 918.69 | 930.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 13:45:00 | 919.10 | 918.69 | 930.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 961.45 | 926.57 | 931.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 961.45 | 926.57 | 931.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 960.05 | 939.08 | 936.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 970.90 | 952.48 | 943.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 952.00 | 955.50 | 946.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 952.00 | 955.50 | 946.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 952.00 | 955.50 | 946.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 938.95 | 955.50 | 946.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 900.65 | 944.53 | 942.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 900.65 | 944.53 | 942.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 837.85 | 923.20 | 933.23 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 945.90 | 910.54 | 910.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 951.90 | 938.88 | 929.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 953.00 | 954.54 | 944.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 954.40 | 955.52 | 946.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 992.70 | 992.20 | 987.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 970.75 | 992.20 | 987.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 976.75 | 989.11 | 986.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 971.85 | 989.11 | 986.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 976.80 | 986.65 | 985.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 976.00 | 986.65 | 985.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 11:15:00 | 977.85 | 984.89 | 985.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 974.00 | 981.23 | 983.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 975.00 | 973.53 | 977.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 975.00 | 973.53 | 977.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 975.00 | 973.53 | 977.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:45:00 | 981.90 | 973.53 | 977.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 972.60 | 973.34 | 976.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 973.45 | 973.34 | 976.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 974.55 | 972.89 | 975.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 976.30 | 972.89 | 975.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 967.95 | 971.91 | 974.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 962.50 | 971.21 | 974.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 10:15:00 | 999.15 | 977.11 | 976.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 999.15 | 977.11 | 976.48 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 09:15:00 | 965.15 | 975.72 | 976.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 961.05 | 972.79 | 975.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 957.00 | 954.62 | 960.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 957.00 | 954.62 | 960.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 957.00 | 954.62 | 960.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 961.10 | 954.62 | 960.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 957.55 | 953.62 | 957.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 957.55 | 953.62 | 957.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 959.60 | 954.82 | 957.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 974.85 | 954.82 | 957.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 987.70 | 961.40 | 960.58 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 973.25 | 978.75 | 979.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 971.90 | 975.52 | 977.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 1002.00 | 964.21 | 966.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1002.00 | 964.21 | 966.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1002.00 | 964.21 | 966.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 1002.00 | 964.21 | 966.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 979.30 | 967.23 | 967.61 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 11:15:00 | 977.80 | 969.34 | 968.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 1053.10 | 986.97 | 976.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 996.00 | 1014.59 | 1000.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 996.00 | 1014.59 | 1000.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 996.00 | 1014.59 | 1000.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 996.00 | 1014.59 | 1000.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1001.00 | 1011.87 | 1000.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 993.35 | 1011.87 | 1000.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1006.40 | 1010.78 | 1001.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 1011.00 | 1007.64 | 1001.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-16 09:15:00 | 1112.10 | 1106.91 | 1075.05 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 1054.60 | 1142.01 | 1151.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 1033.00 | 1104.43 | 1131.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1013.95 | 1006.07 | 1033.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 10:00:00 | 1013.95 | 1006.07 | 1033.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 1040.50 | 1016.10 | 1023.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:30:00 | 1044.95 | 1016.10 | 1023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 1040.25 | 1020.93 | 1024.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 14:30:00 | 1030.00 | 1023.99 | 1025.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 1037.95 | 1017.66 | 1016.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 1037.95 | 1017.66 | 1016.16 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 1010.65 | 1018.46 | 1018.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 12:15:00 | 1007.00 | 1016.17 | 1017.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 1023.00 | 1010.37 | 1013.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1023.00 | 1010.37 | 1013.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1023.00 | 1010.37 | 1013.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 1023.00 | 1010.37 | 1013.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1013.65 | 1011.02 | 1013.78 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 13:15:00 | 1024.45 | 1017.08 | 1016.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 15:15:00 | 1030.00 | 1021.47 | 1018.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1006.75 | 1018.52 | 1017.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1006.75 | 1018.52 | 1017.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1006.75 | 1018.52 | 1017.34 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 1007.25 | 1014.94 | 1015.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 1004.00 | 1011.60 | 1014.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 923.50 | 923.08 | 943.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 925.30 | 923.08 | 943.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 929.75 | 930.63 | 939.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 926.40 | 931.29 | 937.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:00:00 | 926.45 | 930.32 | 936.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 09:30:00 | 923.40 | 925.49 | 932.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:15:00 | 923.30 | 925.80 | 932.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 918.10 | 924.26 | 930.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:30:00 | 916.50 | 922.31 | 929.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 913.50 | 914.51 | 918.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 09:15:00 | 950.75 | 911.75 | 913.12 | SL hit (close>static) qty=1.00 sl=946.80 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 944.20 | 918.24 | 915.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 12:15:00 | 964.80 | 933.96 | 923.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 983.35 | 985.46 | 974.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 983.35 | 985.46 | 974.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 981.55 | 983.54 | 975.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 979.95 | 983.54 | 975.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 976.00 | 980.90 | 976.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 993.15 | 980.90 | 976.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 991.25 | 1001.11 | 1001.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 991.25 | 1001.11 | 1001.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 984.80 | 994.20 | 997.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 973.35 | 968.95 | 975.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 973.35 | 968.95 | 975.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 973.35 | 968.95 | 975.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 969.30 | 969.68 | 974.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:45:00 | 968.85 | 969.78 | 973.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:30:00 | 968.95 | 970.82 | 973.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 986.00 | 975.99 | 975.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 10:15:00 | 986.00 | 975.99 | 975.69 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 978.20 | 980.02 | 980.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 973.80 | 978.77 | 979.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 10:15:00 | 980.00 | 976.44 | 978.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 10:15:00 | 980.00 | 976.44 | 978.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 980.00 | 976.44 | 978.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 980.00 | 976.44 | 978.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 976.75 | 976.50 | 977.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:45:00 | 977.50 | 976.50 | 977.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 977.50 | 976.70 | 977.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:30:00 | 977.20 | 976.70 | 977.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 977.55 | 976.87 | 977.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:45:00 | 976.15 | 976.87 | 977.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 974.10 | 976.32 | 977.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 15:15:00 | 970.15 | 976.32 | 977.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 13:15:00 | 921.64 | 949.80 | 962.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 901.50 | 901.43 | 916.43 | SL hit (close>ema200) qty=0.50 sl=901.43 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 886.80 | 879.85 | 879.69 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 878.00 | 879.80 | 879.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 871.30 | 878.10 | 879.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 879.00 | 873.97 | 876.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 879.00 | 873.97 | 876.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 879.00 | 873.97 | 876.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 879.00 | 873.97 | 876.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 882.40 | 875.65 | 876.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 875.70 | 875.65 | 876.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 877.85 | 875.72 | 876.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 877.85 | 875.72 | 876.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 880.85 | 876.74 | 876.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 880.85 | 876.74 | 876.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 880.25 | 877.44 | 877.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 906.05 | 883.30 | 879.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 915.15 | 917.79 | 909.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 09:45:00 | 918.45 | 917.79 | 909.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 915.80 | 921.17 | 915.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 915.10 | 921.17 | 915.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 921.85 | 921.30 | 916.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 922.45 | 919.21 | 916.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 922.15 | 919.25 | 917.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:15:00 | 928.90 | 918.77 | 917.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 936.30 | 919.25 | 918.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 922.95 | 932.65 | 930.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 920.15 | 928.31 | 928.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 920.15 | 928.31 | 928.44 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 946.25 | 927.57 | 927.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 954.75 | 933.01 | 929.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 940.15 | 940.35 | 935.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 940.15 | 940.35 | 935.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 923.80 | 937.15 | 934.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 923.80 | 937.15 | 934.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 913.30 | 932.38 | 932.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 892.30 | 913.05 | 921.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 14:15:00 | 920.55 | 907.80 | 914.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 14:15:00 | 920.55 | 907.80 | 914.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 920.55 | 907.80 | 914.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:30:00 | 917.50 | 907.80 | 914.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 932.00 | 912.64 | 916.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 929.70 | 912.64 | 916.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 928.45 | 919.22 | 919.00 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 911.25 | 918.95 | 919.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 909.30 | 917.02 | 918.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 906.65 | 903.44 | 908.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 906.65 | 903.44 | 908.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 906.65 | 903.44 | 908.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 906.65 | 903.44 | 908.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 908.20 | 904.39 | 908.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 904.65 | 904.39 | 908.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 903.85 | 904.28 | 908.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:15:00 | 900.95 | 904.28 | 908.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 900.80 | 903.22 | 906.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 13:00:00 | 900.75 | 903.22 | 906.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 900.55 | 903.81 | 905.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 894.95 | 891.98 | 896.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:45:00 | 894.35 | 891.98 | 896.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 933.50 | 900.28 | 899.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 933.50 | 900.28 | 899.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-17 09:15:00 | 970.40 | 919.57 | 908.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 15:15:00 | 942.20 | 943.08 | 928.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-18 09:15:00 | 915.15 | 943.08 | 928.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 920.35 | 938.53 | 927.32 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 902.60 | 925.13 | 925.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 896.65 | 915.42 | 920.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 11:15:00 | 717.05 | 713.76 | 743.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 12:00:00 | 717.05 | 713.76 | 743.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 617.00 | 641.16 | 654.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:00:00 | 613.55 | 635.63 | 650.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 614.70 | 625.51 | 640.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 613.55 | 621.64 | 634.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 671.40 | 637.95 | 637.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 671.40 | 637.95 | 637.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 675.60 | 645.48 | 641.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 660.20 | 660.53 | 653.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 660.20 | 660.53 | 653.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 656.30 | 659.07 | 654.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 644.35 | 659.07 | 654.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 634.45 | 654.15 | 653.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 632.80 | 654.15 | 653.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 626.80 | 648.68 | 650.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 625.20 | 643.98 | 648.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 633.40 | 624.64 | 632.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 14:15:00 | 633.40 | 624.64 | 632.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 633.40 | 624.64 | 632.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 15:00:00 | 633.40 | 624.64 | 632.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 628.00 | 625.31 | 631.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 625.50 | 625.31 | 631.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 594.23 | 606.50 | 617.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 590.80 | 577.27 | 584.08 | SL hit (close>ema200) qty=0.50 sl=577.27 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 583.30 | 579.55 | 579.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 598.70 | 584.53 | 581.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 591.55 | 592.05 | 587.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 591.55 | 592.05 | 587.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 614.25 | 596.47 | 590.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:00:00 | 619.45 | 609.68 | 601.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 12:15:00 | 632.90 | 638.90 | 639.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 12:15:00 | 632.90 | 638.90 | 639.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 14:15:00 | 630.00 | 636.34 | 637.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 605.15 | 600.76 | 608.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 15:00:00 | 605.15 | 600.76 | 608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 610.00 | 602.61 | 608.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 608.10 | 602.61 | 608.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 604.00 | 602.89 | 608.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 599.15 | 603.33 | 606.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:45:00 | 601.50 | 603.25 | 605.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:15:00 | 601.35 | 603.25 | 605.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:45:00 | 601.50 | 602.36 | 605.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 591.65 | 594.50 | 598.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 590.50 | 594.50 | 598.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:45:00 | 591.00 | 593.16 | 596.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 587.35 | 592.18 | 594.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 569.19 | 576.43 | 582.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 571.42 | 576.43 | 582.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 571.28 | 576.43 | 582.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 571.42 | 576.43 | 582.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 11:15:00 | 602.30 | 581.26 | 583.44 | SL hit (close>ema200) qty=0.50 sl=581.26 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 599.95 | 585.00 | 584.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 14:15:00 | 606.85 | 592.48 | 588.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 597.10 | 600.96 | 595.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 597.10 | 600.96 | 595.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 595.10 | 599.79 | 595.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 595.10 | 599.79 | 595.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 592.75 | 598.38 | 595.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 588.85 | 598.38 | 595.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 593.10 | 597.33 | 595.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 592.25 | 597.33 | 595.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 594.85 | 596.83 | 595.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 602.70 | 596.83 | 595.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:30:00 | 597.00 | 596.68 | 595.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 14:00:00 | 596.90 | 596.68 | 595.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 596.25 | 598.33 | 596.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 596.50 | 597.96 | 596.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 596.25 | 597.96 | 596.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 594.95 | 597.36 | 596.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 594.95 | 597.36 | 596.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 593.75 | 596.64 | 596.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-24 14:15:00 | 595.00 | 595.78 | 595.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 595.00 | 595.78 | 595.86 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 596.45 | 595.92 | 595.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 09:15:00 | 606.10 | 597.95 | 596.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 11:15:00 | 604.00 | 605.06 | 602.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 604.00 | 605.06 | 602.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 601.65 | 604.78 | 603.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 600.00 | 604.78 | 603.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 601.00 | 604.02 | 603.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 601.00 | 604.02 | 603.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 600.25 | 602.67 | 602.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:30:00 | 600.30 | 602.67 | 602.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 597.95 | 601.73 | 602.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 591.15 | 599.61 | 601.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 608.85 | 598.76 | 600.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 608.85 | 598.76 | 600.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 608.85 | 598.76 | 600.16 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 622.15 | 603.43 | 602.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 636.05 | 609.96 | 605.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 633.50 | 635.50 | 631.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 633.50 | 635.50 | 631.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 633.70 | 635.14 | 631.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 621.50 | 635.14 | 631.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 619.20 | 631.95 | 630.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 619.10 | 631.95 | 630.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 604.30 | 626.42 | 627.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 601.70 | 615.44 | 622.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 609.30 | 609.23 | 617.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 610.40 | 609.23 | 617.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 628.20 | 610.92 | 613.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 632.20 | 610.92 | 613.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 624.00 | 613.53 | 614.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:15:00 | 619.45 | 613.53 | 614.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 616.45 | 615.07 | 615.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 616.45 | 615.07 | 615.00 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 11:15:00 | 613.75 | 614.80 | 614.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 609.90 | 613.82 | 614.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 14:15:00 | 613.85 | 613.38 | 614.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 14:15:00 | 613.85 | 613.38 | 614.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 613.85 | 613.38 | 614.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 15:00:00 | 613.85 | 613.38 | 614.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 610.00 | 612.70 | 613.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 602.55 | 612.70 | 613.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 10:45:00 | 607.95 | 610.43 | 612.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 11:30:00 | 606.45 | 609.52 | 611.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:00:00 | 605.90 | 609.52 | 611.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 594.50 | 603.00 | 607.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 10:45:00 | 589.60 | 600.22 | 605.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 577.55 | 587.54 | 597.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 576.13 | 587.54 | 597.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 575.60 | 587.54 | 597.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 572.42 | 582.01 | 593.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 578.90 | 576.91 | 585.11 | SL hit (close>ema200) qty=0.50 sl=576.91 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 14:15:00 | 592.20 | 580.66 | 579.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 600.90 | 586.87 | 583.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 588.60 | 593.16 | 589.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 588.60 | 593.16 | 589.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 588.60 | 593.16 | 589.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 588.60 | 593.16 | 589.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 590.45 | 592.62 | 589.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 591.85 | 592.62 | 589.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 590.60 | 592.21 | 589.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 591.25 | 592.21 | 589.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 590.00 | 591.77 | 589.34 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 572.95 | 585.81 | 587.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 568.10 | 580.13 | 584.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 582.15 | 578.27 | 582.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 582.15 | 578.27 | 582.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 582.15 | 578.27 | 582.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 582.15 | 578.27 | 582.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 581.95 | 579.01 | 582.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 09:15:00 | 573.95 | 579.01 | 582.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 574.30 | 573.52 | 577.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 545.25 | 556.45 | 566.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 545.58 | 556.45 | 566.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 516.56 | 532.60 | 547.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 12:15:00 | 538.15 | 531.78 | 531.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 541.20 | 534.65 | 533.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 537.80 | 539.03 | 535.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 537.80 | 539.03 | 535.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 537.80 | 539.03 | 535.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 535.70 | 539.03 | 535.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 530.85 | 537.39 | 535.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 530.85 | 537.39 | 535.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 537.85 | 537.48 | 535.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 542.20 | 538.43 | 536.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 518.10 | 534.61 | 534.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 518.10 | 534.61 | 534.95 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 534.80 | 528.37 | 527.55 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 14:15:00 | 527.30 | 529.37 | 529.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 525.10 | 528.04 | 528.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 518.40 | 517.96 | 521.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 518.40 | 517.96 | 521.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 509.60 | 502.08 | 505.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 509.25 | 502.08 | 505.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 508.10 | 503.28 | 505.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 508.75 | 503.28 | 505.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 502.80 | 504.11 | 505.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 500.50 | 503.73 | 505.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 475.47 | 482.40 | 491.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 479.90 | 479.29 | 487.50 | SL hit (close>ema200) qty=0.50 sl=479.29 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 492.10 | 484.19 | 483.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 495.45 | 488.60 | 486.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 497.50 | 498.05 | 494.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:00:00 | 497.50 | 498.05 | 494.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 498.30 | 500.94 | 497.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 10:30:00 | 503.20 | 501.51 | 498.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 14:15:00 | 492.15 | 499.47 | 500.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 492.15 | 499.47 | 500.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 482.85 | 494.95 | 497.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 451.05 | 449.86 | 462.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 451.05 | 449.86 | 462.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 458.25 | 452.02 | 459.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 458.25 | 452.02 | 459.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 458.75 | 453.36 | 459.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:00:00 | 458.75 | 453.36 | 459.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 454.35 | 453.56 | 459.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 458.15 | 453.56 | 459.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 462.85 | 455.78 | 458.73 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 470.45 | 461.34 | 460.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 476.25 | 464.33 | 462.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 520.55 | 523.49 | 511.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 520.55 | 523.49 | 511.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 513.75 | 520.50 | 513.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 513.75 | 520.50 | 513.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 511.00 | 518.60 | 513.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 519.50 | 518.60 | 513.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 541.95 | 523.27 | 516.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:00:00 | 548.95 | 528.41 | 519.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-21 13:15:00 | 603.85 | 591.38 | 580.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 613.05 | 620.62 | 621.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 09:15:00 | 608.15 | 618.12 | 620.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 606.00 | 602.26 | 607.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 606.00 | 602.26 | 607.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 606.00 | 602.26 | 607.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 608.65 | 602.26 | 607.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 603.30 | 602.46 | 607.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:45:00 | 606.50 | 602.46 | 607.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 608.10 | 603.59 | 607.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 608.10 | 603.59 | 607.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 607.00 | 604.27 | 607.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 610.55 | 604.27 | 607.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 603.10 | 604.04 | 607.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 567.35 | 605.99 | 606.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 538.98 | 596.87 | 602.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 605.15 | 586.96 | 594.73 | SL hit (close>ema200) qty=0.50 sl=586.96 alert=retest2 |

### Cycle 57 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 605.55 | 597.81 | 597.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 608.00 | 599.85 | 598.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 599.65 | 602.39 | 600.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 599.65 | 602.39 | 600.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 599.65 | 602.39 | 600.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 10:00:00 | 599.65 | 602.39 | 600.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 606.55 | 603.22 | 600.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:15:00 | 612.50 | 603.22 | 600.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 611.90 | 608.93 | 605.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:45:00 | 608.75 | 608.83 | 605.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-23 11:15:00 | 627.10 | 630.51 | 630.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 11:15:00 | 627.10 | 630.51 | 630.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 617.55 | 626.77 | 628.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 641.50 | 626.58 | 627.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 13:15:00 | 641.50 | 626.58 | 627.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 641.50 | 626.58 | 627.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 13:45:00 | 649.40 | 626.58 | 627.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 630.00 | 627.27 | 627.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 15:15:00 | 629.10 | 627.27 | 627.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 632.60 | 619.99 | 619.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 632.60 | 619.99 | 619.34 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 616.20 | 620.63 | 621.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 609.00 | 617.33 | 619.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 14:15:00 | 615.70 | 612.02 | 614.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 615.70 | 612.02 | 614.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 615.70 | 612.02 | 614.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 615.70 | 612.02 | 614.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 615.65 | 612.75 | 614.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 622.70 | 612.75 | 614.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 623.90 | 614.98 | 615.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 625.40 | 614.98 | 615.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 623.25 | 616.63 | 616.42 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 607.45 | 615.62 | 616.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 602.00 | 612.90 | 614.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 620.70 | 600.82 | 604.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 620.70 | 600.82 | 604.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 620.70 | 600.82 | 604.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 620.70 | 600.82 | 604.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 618.55 | 609.26 | 608.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 622.65 | 615.26 | 611.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 645.00 | 649.58 | 644.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 645.00 | 649.58 | 644.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 653.40 | 650.35 | 644.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:00:00 | 671.50 | 654.60 | 647.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 672.80 | 702.20 | 704.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 672.80 | 702.20 | 704.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 668.05 | 695.37 | 700.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 654.30 | 654.13 | 663.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 13:45:00 | 653.70 | 654.13 | 663.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 659.95 | 656.12 | 661.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 655.00 | 656.84 | 660.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:00:00 | 654.15 | 656.30 | 659.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 653.90 | 651.89 | 654.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 657.25 | 655.68 | 655.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 662.55 | 657.05 | 656.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 672.00 | 679.42 | 672.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 672.00 | 679.42 | 672.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 670.85 | 677.71 | 672.50 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 664.65 | 669.41 | 669.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 651.85 | 665.90 | 668.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 642.40 | 633.99 | 642.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 641.20 | 633.99 | 642.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 637.50 | 635.53 | 641.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 633.50 | 635.96 | 639.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 633.00 | 635.96 | 639.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:45:00 | 632.35 | 634.49 | 638.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 643.80 | 629.86 | 628.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 673.90 | 638.67 | 632.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 679.10 | 680.27 | 664.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 679.10 | 680.27 | 664.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 668.35 | 676.26 | 670.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 668.35 | 676.26 | 670.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 667.50 | 674.51 | 670.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:30:00 | 667.00 | 674.51 | 670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 671.50 | 673.35 | 671.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:00:00 | 671.50 | 673.35 | 671.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 672.65 | 673.21 | 671.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 671.05 | 673.21 | 671.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 672.20 | 673.01 | 671.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 673.30 | 673.01 | 671.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 669.55 | 672.32 | 671.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 674.45 | 672.74 | 671.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 11:15:00 | 741.90 | 724.96 | 711.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 727.00 | 735.23 | 735.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 720.45 | 730.15 | 732.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 722.70 | 720.45 | 724.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 722.70 | 720.45 | 724.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 722.05 | 720.77 | 724.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 719.65 | 720.79 | 724.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 719.80 | 721.03 | 724.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 727.15 | 722.06 | 723.99 | SL hit (close>static) qty=1.00 sl=725.60 alert=retest2 |

### Cycle 69 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 734.00 | 725.87 | 725.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 739.45 | 728.58 | 726.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 12:15:00 | 745.50 | 748.04 | 743.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:45:00 | 745.50 | 748.04 | 743.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 754.05 | 749.19 | 745.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 759.50 | 750.19 | 745.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:45:00 | 771.20 | 754.51 | 749.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 762.40 | 771.62 | 771.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 762.40 | 771.62 | 771.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 759.50 | 769.20 | 770.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 13:15:00 | 727.80 | 717.45 | 733.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 727.80 | 717.45 | 733.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 696.25 | 705.69 | 716.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 688.30 | 698.77 | 707.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 653.88 | 668.74 | 681.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 640.95 | 637.60 | 646.31 | SL hit (close>ema200) qty=0.50 sl=637.60 alert=retest2 |

### Cycle 71 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 652.15 | 643.45 | 642.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 656.00 | 648.50 | 645.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 658.75 | 658.76 | 654.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:30:00 | 658.50 | 658.76 | 654.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 656.90 | 658.39 | 654.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 655.15 | 658.39 | 654.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 656.30 | 657.67 | 655.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 655.60 | 657.67 | 655.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 653.50 | 656.83 | 654.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 653.50 | 656.83 | 654.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 651.90 | 655.85 | 654.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 651.90 | 655.85 | 654.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 641.70 | 652.24 | 653.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 639.40 | 649.67 | 651.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 639.05 | 633.93 | 638.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:15:00 | 642.70 | 633.93 | 638.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 644.10 | 635.96 | 639.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 644.10 | 635.96 | 639.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 652.30 | 639.23 | 640.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 652.30 | 639.23 | 640.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 649.00 | 642.59 | 641.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 656.25 | 645.32 | 643.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 662.55 | 665.49 | 659.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 662.55 | 665.49 | 659.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 659.30 | 663.72 | 660.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 656.30 | 663.72 | 660.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 658.20 | 662.62 | 659.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 658.20 | 662.62 | 659.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 660.50 | 662.19 | 659.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 662.15 | 662.19 | 660.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 663.80 | 660.99 | 660.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 662.65 | 661.15 | 660.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 655.55 | 660.02 | 660.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 650.00 | 655.27 | 657.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 648.40 | 648.30 | 651.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 648.40 | 648.30 | 651.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 652.60 | 647.57 | 649.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 653.35 | 647.57 | 649.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 668.55 | 651.77 | 651.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 683.00 | 658.01 | 654.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 696.15 | 697.80 | 687.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 696.15 | 697.80 | 687.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 697.45 | 699.10 | 693.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 698.95 | 698.18 | 693.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 690.00 | 696.55 | 693.48 | SL hit (close<static) qty=1.00 sl=692.10 alert=retest2 |

### Cycle 76 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 687.50 | 691.65 | 691.90 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 700.00 | 693.32 | 692.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 14:15:00 | 702.65 | 695.19 | 693.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 702.50 | 703.07 | 699.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 702.50 | 703.07 | 699.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 736.65 | 740.92 | 735.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 736.65 | 740.92 | 735.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 732.60 | 739.26 | 735.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 733.60 | 739.26 | 735.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 733.45 | 738.10 | 734.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 733.80 | 738.10 | 734.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 749.00 | 739.69 | 736.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 753.40 | 743.87 | 739.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 753.90 | 746.30 | 741.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 755.60 | 750.83 | 744.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 754.55 | 752.25 | 747.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 747.25 | 751.35 | 748.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 747.25 | 751.35 | 748.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 747.10 | 750.50 | 748.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 747.10 | 750.50 | 748.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 748.00 | 750.00 | 748.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 748.25 | 750.00 | 748.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 741.50 | 748.30 | 747.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 741.50 | 748.30 | 747.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 739.40 | 746.52 | 746.82 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 752.00 | 747.14 | 746.87 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 744.10 | 746.53 | 746.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 15:15:00 | 744.00 | 745.55 | 746.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 746.50 | 745.74 | 746.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 746.40 | 745.74 | 746.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 747.15 | 746.02 | 746.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 747.15 | 746.02 | 746.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 744.10 | 745.64 | 746.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:15:00 | 743.00 | 745.76 | 746.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 754.90 | 746.95 | 746.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 754.90 | 746.95 | 746.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 762.50 | 751.30 | 748.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 749.00 | 754.37 | 751.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 749.00 | 754.37 | 751.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 746.00 | 752.69 | 751.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 746.90 | 752.69 | 751.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 754.40 | 752.64 | 751.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 759.50 | 753.14 | 751.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 756.70 | 754.03 | 752.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 759.75 | 753.83 | 752.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:30:00 | 757.85 | 758.28 | 756.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 760.50 | 762.24 | 759.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 766.30 | 762.24 | 759.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 765.10 | 767.11 | 764.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 755.50 | 763.20 | 763.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 754.55 | 761.47 | 762.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 764.85 | 761.17 | 762.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 764.85 | 761.17 | 762.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 758.00 | 760.54 | 761.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:00:00 | 750.95 | 758.62 | 760.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 765.60 | 760.35 | 760.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 765.60 | 760.35 | 760.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 773.85 | 763.05 | 761.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 804.80 | 806.71 | 796.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 811.00 | 806.71 | 796.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 811.75 | 806.77 | 797.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 809.95 | 803.57 | 799.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 785.00 | 798.32 | 799.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 775.00 | 787.00 | 792.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 772.50 | 770.93 | 778.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 11:00:00 | 772.50 | 770.93 | 778.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 744.75 | 760.95 | 770.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:45:00 | 740.90 | 754.33 | 765.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 12:30:00 | 743.15 | 752.91 | 763.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:30:00 | 741.85 | 750.09 | 761.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 777.10 | 746.67 | 745.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 780.70 | 764.00 | 755.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 748.95 | 763.23 | 756.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 740.65 | 763.23 | 756.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 743.60 | 759.30 | 755.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:30:00 | 740.70 | 759.30 | 755.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 743.95 | 751.80 | 752.54 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 765.00 | 752.73 | 752.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 11:15:00 | 775.80 | 759.43 | 755.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 764.00 | 771.75 | 765.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 757.50 | 771.75 | 765.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 796.70 | 776.74 | 768.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:45:00 | 835.35 | 797.25 | 782.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 846.80 | 820.23 | 807.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-31 09:15:00 | 918.89 | 871.24 | 842.94 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1036.20 | 1067.51 | 1070.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1031.00 | 1048.58 | 1059.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1018.45 | 1004.04 | 1017.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 1021.00 | 1004.04 | 1017.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1019.45 | 1007.12 | 1017.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1019.45 | 1007.12 | 1017.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1015.00 | 1008.70 | 1017.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 976.85 | 1008.70 | 1017.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 12:15:00 | 928.01 | 969.58 | 994.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 938.00 | 933.15 | 960.79 | SL hit (close>ema200) qty=0.50 sl=933.15 alert=retest2 |

### Cycle 89 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 933.70 | 928.33 | 927.79 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 914.60 | 926.09 | 926.90 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 937.55 | 927.03 | 926.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 14:15:00 | 943.15 | 932.27 | 929.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 931.45 | 938.79 | 934.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 931.45 | 938.79 | 934.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 932.60 | 937.55 | 934.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:15:00 | 926.50 | 937.55 | 934.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 932.00 | 934.11 | 933.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 934.45 | 934.11 | 933.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 927.95 | 932.88 | 933.09 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 09:15:00 | 945.00 | 933.67 | 933.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 10:15:00 | 945.60 | 936.06 | 934.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 930.00 | 935.69 | 934.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 930.00 | 935.69 | 934.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 933.45 | 935.24 | 934.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 928.10 | 935.24 | 934.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 928.00 | 933.79 | 933.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 901.55 | 927.34 | 931.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 926.00 | 923.72 | 927.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 926.00 | 923.72 | 927.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 947.30 | 927.29 | 928.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 940.90 | 927.29 | 928.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 947.00 | 931.23 | 929.97 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 920.25 | 932.50 | 932.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 907.50 | 916.58 | 921.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 917.70 | 915.71 | 919.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 13:30:00 | 910.10 | 915.01 | 918.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 912.50 | 914.85 | 918.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:15:00 | 864.60 | 877.95 | 886.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:15:00 | 866.88 | 877.95 | 886.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-29 12:15:00 | 819.09 | 829.16 | 842.44 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 97 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 846.05 | 826.05 | 824.94 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 825.95 | 832.63 | 833.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 817.00 | 829.08 | 831.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 813.30 | 809.32 | 814.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 814.10 | 809.32 | 814.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 809.50 | 809.35 | 814.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 808.25 | 809.35 | 814.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 817.80 | 810.85 | 813.78 | SL hit (close>static) qty=1.00 sl=815.00 alert=retest2 |

### Cycle 99 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 813.90 | 794.63 | 793.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 822.40 | 807.35 | 801.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 852.70 | 857.97 | 838.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 852.70 | 857.97 | 838.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 854.10 | 859.04 | 847.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 851.15 | 859.04 | 847.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 849.50 | 857.13 | 848.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 849.10 | 857.13 | 848.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 849.60 | 855.63 | 848.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 849.85 | 855.63 | 848.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 848.50 | 854.20 | 848.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:30:00 | 846.65 | 854.20 | 848.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 852.05 | 853.77 | 848.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 848.45 | 853.77 | 848.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 847.35 | 852.49 | 848.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 847.35 | 852.49 | 848.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 843.05 | 850.60 | 847.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 837.70 | 850.60 | 847.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 827.00 | 845.88 | 846.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 814.85 | 839.67 | 843.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 820.60 | 818.81 | 829.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 10:00:00 | 820.60 | 818.81 | 829.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 810.55 | 817.15 | 827.61 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 858.50 | 832.57 | 831.61 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 830.00 | 831.94 | 832.14 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 843.70 | 834.29 | 833.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 866.25 | 842.18 | 837.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 836.40 | 843.81 | 840.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 834.90 | 843.81 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 845.50 | 844.15 | 840.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 861.00 | 844.15 | 840.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 850.20 | 847.29 | 842.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 839.90 | 858.78 | 859.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 839.90 | 858.78 | 859.84 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 875.00 | 850.32 | 849.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 882.10 | 856.68 | 852.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 888.55 | 889.46 | 878.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 903.70 | 889.46 | 878.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 921.30 | 924.59 | 919.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 925.25 | 924.59 | 919.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 918.65 | 923.40 | 919.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 918.65 | 923.40 | 919.73 | SL hit (close<ema400) qty=1.00 sl=919.73 alert=retest1 |

### Cycle 106 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 902.30 | 914.33 | 915.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 877.80 | 902.48 | 909.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 889.05 | 885.37 | 892.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 889.05 | 885.37 | 892.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 883.05 | 885.07 | 890.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 10:15:00 | 879.55 | 885.07 | 890.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 910.35 | 890.20 | 889.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 910.35 | 890.20 | 889.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 918.85 | 907.59 | 900.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 13:15:00 | 909.75 | 911.02 | 904.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 14:00:00 | 909.75 | 911.02 | 904.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 896.55 | 908.13 | 904.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 896.55 | 908.13 | 904.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 891.90 | 904.88 | 902.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 903.30 | 902.72 | 902.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 897.65 | 902.72 | 902.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 894.45 | 901.06 | 901.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 894.45 | 901.06 | 901.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 885.55 | 897.96 | 900.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 895.60 | 889.96 | 894.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 895.60 | 889.96 | 894.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 892.15 | 890.40 | 894.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:30:00 | 896.45 | 890.40 | 894.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 886.95 | 889.71 | 893.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:00:00 | 885.95 | 888.96 | 892.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:30:00 | 884.15 | 887.79 | 891.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 900.50 | 890.12 | 892.18 | SL hit (close>static) qty=1.00 sl=894.85 alert=retest2 |

### Cycle 109 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 909.25 | 893.95 | 893.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 912.00 | 901.70 | 897.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 14:15:00 | 915.95 | 917.09 | 909.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 915.95 | 917.09 | 909.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 917.00 | 917.08 | 911.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 921.80 | 918.26 | 912.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 09:30:00 | 938.00 | 920.82 | 915.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:45:00 | 924.15 | 941.03 | 935.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-05 09:15:00 | 1013.98 | 990.83 | 967.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 944.50 | 986.32 | 987.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 941.95 | 965.69 | 977.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 947.25 | 923.39 | 934.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 947.25 | 923.39 | 934.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 954.10 | 929.53 | 936.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:15:00 | 954.25 | 929.53 | 936.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 953.15 | 934.26 | 937.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:30:00 | 946.80 | 939.14 | 939.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 899.46 | 923.98 | 929.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 927.95 | 923.98 | 929.92 | SL hit (close>static) qty=0.50 sl=923.98 alert=retest2 |

### Cycle 111 — BUY (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 13:15:00 | 951.00 | 934.71 | 933.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-16 14:15:00 | 986.95 | 945.15 | 938.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 1016.75 | 1024.98 | 993.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:45:00 | 1019.90 | 1024.98 | 993.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1034.50 | 1020.26 | 1005.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1076.90 | 1029.57 | 1017.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 1065.10 | 1037.58 | 1022.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1007.35 | 1027.65 | 1030.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1007.35 | 1027.65 | 1030.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 09:15:00 | 960.80 | 1010.17 | 1021.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 991.50 | 985.97 | 1002.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 991.50 | 985.97 | 1002.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1022.00 | 993.87 | 1003.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1022.00 | 993.87 | 1003.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1023.40 | 999.78 | 1005.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 1019.70 | 1003.82 | 1006.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 1019.40 | 1003.82 | 1006.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 13:15:00 | 1020.00 | 1010.10 | 1009.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1020.00 | 1010.10 | 1009.19 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 1000.00 | 1007.77 | 1008.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 949.95 | 996.21 | 1002.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 982.75 | 968.80 | 981.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:30:00 | 997.30 | 968.80 | 981.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 974.70 | 969.98 | 980.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 15:00:00 | 965.80 | 973.77 | 979.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 993.00 | 977.33 | 980.17 | SL hit (close>static) qty=1.00 sl=982.90 alert=retest2 |

### Cycle 115 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1008.85 | 985.98 | 983.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 1015.20 | 991.83 | 986.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 985.35 | 999.37 | 992.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 985.35 | 999.37 | 992.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1000.70 | 999.63 | 993.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 1013.00 | 1003.14 | 995.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 1013.25 | 1014.97 | 1005.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 991.40 | 1000.11 | 1000.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 13:15:00 | 991.40 | 1000.11 | 1000.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 14:15:00 | 981.00 | 996.29 | 998.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1008.90 | 983.83 | 987.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 1008.90 | 983.83 | 987.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 993.25 | 985.71 | 988.38 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 999.00 | 990.88 | 990.42 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 11:15:00 | 988.05 | 990.14 | 990.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 12:15:00 | 986.00 | 989.31 | 989.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 15:15:00 | 974.40 | 974.37 | 980.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 955.45 | 974.37 | 980.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 953.20 | 953.54 | 963.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 966.40 | 956.11 | 964.16 | SL hit (close>ema400) qty=1.00 sl=964.16 alert=retest1 |

### Cycle 119 — BUY (started 2026-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 15:15:00 | 972.70 | 968.73 | 968.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 989.10 | 972.81 | 970.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 980.80 | 981.90 | 976.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 980.80 | 981.90 | 976.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1008.45 | 1011.70 | 1005.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1019.90 | 1011.70 | 1005.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 13:30:00 | 1019.30 | 1059.16 | 1055.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:30:00 | 1016.10 | 1054.33 | 1053.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 1037.50 | 1050.96 | 1052.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 09:15:00 | 998.00 | 1040.37 | 1047.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1048.80 | 1017.07 | 1027.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 1067.00 | 1017.07 | 1027.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1078.20 | 1029.30 | 1032.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1078.20 | 1029.30 | 1032.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 1058.70 | 1035.18 | 1034.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1117.50 | 1064.36 | 1049.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 1113.40 | 1117.45 | 1093.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 12:30:00 | 1124.05 | 1118.58 | 1097.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 14:30:00 | 1130.45 | 1121.47 | 1102.81 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1122.20 | 1122.02 | 1106.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1094.90 | 1111.11 | 1107.98 | SL hit (close<ema400) qty=1.00 sl=1107.98 alert=retest1 |

### Cycle 122 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 1089.50 | 1103.52 | 1104.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 1074.10 | 1089.04 | 1096.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1074.10 | 1074.04 | 1082.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 1080.00 | 1074.04 | 1082.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1088.60 | 1078.43 | 1081.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 12:45:00 | 1081.40 | 1081.49 | 1082.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 13:30:00 | 1080.50 | 1080.59 | 1081.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 14:45:00 | 1079.10 | 1079.85 | 1081.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 09:15:00 | 887.20 | 2024-05-21 11:15:00 | 896.75 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-05-21 10:00:00 | 894.55 | 2024-05-21 11:15:00 | 896.75 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-06-24 09:15:00 | 962.50 | 2024-06-24 10:15:00 | 999.15 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2024-07-10 15:15:00 | 1011.00 | 2024-07-16 09:15:00 | 1112.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-25 14:30:00 | 1030.00 | 2024-07-30 09:15:00 | 1037.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-08-08 13:15:00 | 926.40 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-08-08 14:00:00 | 926.45 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-08-09 09:30:00 | 923.40 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-08-09 11:15:00 | 923.30 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-08-09 12:30:00 | 916.50 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-08-13 09:15:00 | 913.50 | 2024-08-14 09:15:00 | 950.75 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-08-21 09:15:00 | 993.15 | 2024-08-27 09:15:00 | 991.25 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-08-30 11:30:00 | 969.30 | 2024-09-02 10:15:00 | 986.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-08-30 13:45:00 | 968.85 | 2024-09-02 10:15:00 | 986.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-08-30 14:30:00 | 968.95 | 2024-09-02 10:15:00 | 986.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-09-05 15:15:00 | 970.15 | 2024-09-06 13:15:00 | 921.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 15:15:00 | 970.15 | 2024-09-10 13:15:00 | 901.50 | STOP_HIT | 0.50 | 7.08% |
| BUY | retest2 | 2024-09-27 09:15:00 | 922.45 | 2024-10-03 11:15:00 | 920.15 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-09-27 10:15:00 | 922.15 | 2024-10-03 11:15:00 | 920.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-09-27 13:15:00 | 928.90 | 2024-10-03 11:15:00 | 920.15 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-30 09:15:00 | 936.30 | 2024-10-03 11:15:00 | 920.15 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-10-14 10:15:00 | 900.95 | 2024-10-16 14:15:00 | 933.50 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2024-10-14 12:30:00 | 900.80 | 2024-10-16 14:15:00 | 933.50 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-10-14 13:00:00 | 900.75 | 2024-10-16 14:15:00 | 933.50 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2024-10-15 10:00:00 | 900.55 | 2024-10-16 14:15:00 | 933.50 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2024-11-04 11:00:00 | 613.55 | 2024-11-06 09:15:00 | 671.40 | STOP_HIT | 1.00 | -9.43% |
| SELL | retest2 | 2024-11-04 15:00:00 | 614.70 | 2024-11-06 09:15:00 | 671.40 | STOP_HIT | 1.00 | -9.22% |
| SELL | retest2 | 2024-11-05 11:15:00 | 613.55 | 2024-11-06 09:15:00 | 671.40 | STOP_HIT | 1.00 | -9.43% |
| SELL | retest2 | 2024-11-12 09:15:00 | 625.50 | 2024-11-13 09:15:00 | 594.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:15:00 | 625.50 | 2024-11-19 09:15:00 | 590.80 | STOP_HIT | 0.50 | 5.55% |
| BUY | retest2 | 2024-11-27 12:00:00 | 619.45 | 2024-12-05 12:15:00 | 632.90 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2024-12-12 09:30:00 | 599.15 | 2024-12-19 09:15:00 | 569.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 10:45:00 | 601.50 | 2024-12-19 09:15:00 | 571.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 11:15:00 | 601.35 | 2024-12-19 09:15:00 | 571.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 11:45:00 | 601.50 | 2024-12-19 09:15:00 | 571.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 599.15 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 0.50 | -0.53% |
| SELL | retest2 | 2024-12-12 10:45:00 | 601.50 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-12-12 11:15:00 | 601.35 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 0.50 | -0.16% |
| SELL | retest2 | 2024-12-12 11:45:00 | 601.50 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-12-16 10:15:00 | 590.50 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-12-16 11:45:00 | 591.00 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-12-17 09:30:00 | 587.35 | 2024-12-19 11:15:00 | 602.30 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-12-23 11:15:00 | 602.70 | 2024-12-24 14:15:00 | 595.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-12-23 13:30:00 | 597.00 | 2024-12-24 14:15:00 | 595.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-12-23 14:00:00 | 596.90 | 2024-12-24 14:15:00 | 595.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-12-24 09:30:00 | 596.25 | 2024-12-24 14:15:00 | 595.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-01-08 11:15:00 | 619.45 | 2025-01-09 10:15:00 | 616.45 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-01-10 09:15:00 | 602.55 | 2025-01-13 14:15:00 | 577.55 | PARTIAL | 0.50 | 4.15% |
| SELL | retest2 | 2025-01-10 10:45:00 | 607.95 | 2025-01-13 14:15:00 | 576.13 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2025-01-10 11:30:00 | 606.45 | 2025-01-13 14:15:00 | 575.60 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-01-10 12:00:00 | 605.90 | 2025-01-14 09:15:00 | 572.42 | PARTIAL | 0.50 | 5.53% |
| SELL | retest2 | 2025-01-10 09:15:00 | 602.55 | 2025-01-14 15:15:00 | 578.90 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-01-10 10:45:00 | 607.95 | 2025-01-14 15:15:00 | 578.90 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2025-01-10 11:30:00 | 606.45 | 2025-01-14 15:15:00 | 578.90 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-01-10 12:00:00 | 605.90 | 2025-01-14 15:15:00 | 578.90 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-01-13 10:45:00 | 589.60 | 2025-01-17 14:15:00 | 592.20 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-01-23 09:15:00 | 573.95 | 2025-01-27 09:15:00 | 545.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 574.30 | 2025-01-27 09:15:00 | 545.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 09:15:00 | 573.95 | 2025-01-28 09:15:00 | 516.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 574.30 | 2025-01-28 09:15:00 | 516.87 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-01 15:00:00 | 542.20 | 2025-02-03 09:15:00 | 518.10 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-02-13 15:15:00 | 500.50 | 2025-02-17 09:15:00 | 475.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 500.50 | 2025-02-17 12:15:00 | 479.90 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-02-24 10:30:00 | 503.20 | 2025-02-25 14:15:00 | 492.15 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-03-11 11:00:00 | 548.95 | 2025-03-21 13:15:00 | 603.85 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 567.35 | 2025-04-07 09:15:00 | 538.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 567.35 | 2025-04-07 13:15:00 | 605.15 | STOP_HIT | 0.50 | -6.66% |
| SELL | retest2 | 2025-04-07 15:15:00 | 593.00 | 2025-04-08 12:15:00 | 605.55 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-04-08 09:30:00 | 594.95 | 2025-04-08 12:15:00 | 605.55 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-04-09 11:15:00 | 612.50 | 2025-04-23 11:15:00 | 627.10 | STOP_HIT | 1.00 | 2.38% |
| BUY | retest2 | 2025-04-11 09:15:00 | 611.90 | 2025-04-23 11:15:00 | 627.10 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2025-04-11 09:45:00 | 608.75 | 2025-04-23 11:15:00 | 627.10 | STOP_HIT | 1.00 | 3.01% |
| SELL | retest2 | 2025-04-25 15:15:00 | 629.10 | 2025-05-05 09:15:00 | 632.60 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-20 10:00:00 | 671.50 | 2025-05-30 14:15:00 | 672.80 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-06-05 13:30:00 | 655.00 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-05 15:00:00 | 654.15 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-09 10:45:00 | 653.90 | 2025-06-10 10:15:00 | 657.25 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-06-18 09:30:00 | 633.50 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-18 10:15:00 | 633.00 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-06-18 11:45:00 | 632.35 | 2025-06-23 09:15:00 | 643.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-06-27 10:00:00 | 674.45 | 2025-07-04 11:15:00 | 741.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-14 13:45:00 | 719.65 | 2025-07-15 09:15:00 | 727.15 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-07-14 15:15:00 | 719.80 | 2025-07-15 09:15:00 | 727.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-18 11:15:00 | 759.50 | 2025-07-24 11:15:00 | 762.40 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-07-18 13:45:00 | 771.20 | 2025-07-24 11:15:00 | 762.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-07-31 10:00:00 | 688.30 | 2025-08-01 15:15:00 | 653.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 10:00:00 | 688.30 | 2025-08-06 12:15:00 | 640.95 | STOP_HIT | 0.50 | 6.88% |
| BUY | retest2 | 2025-08-22 13:00:00 | 662.15 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-25 10:15:00 | 663.80 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-25 11:15:00 | 662.65 | 2025-08-26 09:15:00 | 655.55 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-04 14:30:00 | 698.95 | 2025-09-04 15:15:00 | 690.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-15 13:15:00 | 753.40 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-09-15 15:15:00 | 753.90 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-09-16 10:30:00 | 755.60 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-09-16 14:15:00 | 754.55 | 2025-09-17 14:15:00 | 739.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-09-19 14:15:00 | 743.00 | 2025-09-22 09:15:00 | 754.90 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-24 10:15:00 | 759.50 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-09-24 12:45:00 | 756.70 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-25 09:15:00 | 759.75 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-26 09:30:00 | 757.85 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-29 09:15:00 | 766.30 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-30 09:30:00 | 765.10 | 2025-09-30 12:15:00 | 755.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-01 12:00:00 | 750.95 | 2025-10-03 11:15:00 | 765.60 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-08 09:15:00 | 811.00 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-10-08 10:15:00 | 811.75 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-10-09 09:15:00 | 809.95 | 2025-10-13 09:15:00 | 785.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-10-16 11:45:00 | 740.90 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2025-10-16 12:30:00 | 743.15 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-10-16 13:30:00 | 741.85 | 2025-10-20 12:15:00 | 777.10 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2025-10-28 13:45:00 | 835.35 | 2025-10-31 09:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:30:00 | 846.80 | 2025-10-31 10:15:00 | 931.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 976.85 | 2025-11-26 12:15:00 | 928.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 976.85 | 2025-11-27 11:15:00 | 938.00 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-12-17 13:30:00 | 910.10 | 2025-12-23 13:15:00 | 864.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 15:15:00 | 912.50 | 2025-12-23 13:15:00 | 866.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 13:30:00 | 910.10 | 2025-12-29 12:15:00 | 819.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-17 15:15:00 | 912.50 | 2025-12-29 12:15:00 | 821.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 808.25 | 2026-01-07 14:15:00 | 817.80 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-01-08 09:15:00 | 808.50 | 2026-01-12 09:15:00 | 768.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 808.50 | 2026-01-12 12:15:00 | 794.05 | STOP_HIT | 0.50 | 1.79% |
| BUY | retest2 | 2026-01-28 09:15:00 | 861.00 | 2026-02-01 12:15:00 | 839.90 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-28 10:30:00 | 850.20 | 2026-02-01 12:15:00 | 839.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-02-06 09:15:00 | 903.70 | 2026-02-12 10:15:00 | 918.65 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2026-02-17 10:15:00 | 879.55 | 2026-02-18 09:15:00 | 910.35 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-02-20 09:30:00 | 903.30 | 2026-02-20 10:15:00 | 894.45 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-02-20 10:15:00 | 897.65 | 2026-02-20 10:15:00 | 894.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-23 14:00:00 | 885.95 | 2026-02-24 09:15:00 | 900.50 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-23 14:30:00 | 884.15 | 2026-02-24 09:15:00 | 900.50 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-02-26 10:30:00 | 921.80 | 2026-03-05 09:15:00 | 1013.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-27 09:30:00 | 938.00 | 2026-03-05 09:15:00 | 1031.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:45:00 | 924.15 | 2026-03-05 09:15:00 | 1016.57 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 946.80 | 2026-03-16 09:15:00 | 899.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 14:30:00 | 946.80 | 2026-03-16 09:15:00 | 927.95 | STOP_HIT | 0.50 | 1.99% |
| BUY | retest2 | 2026-03-20 09:30:00 | 1076.90 | 2026-03-23 14:15:00 | 1007.35 | STOP_HIT | 1.00 | -6.46% |
| BUY | retest2 | 2026-03-20 10:45:00 | 1065.10 | 2026-03-23 14:15:00 | 1007.35 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2026-03-25 11:30:00 | 1019.70 | 2026-03-25 13:15:00 | 1020.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-03-25 12:15:00 | 1019.40 | 2026-03-25 13:15:00 | 1020.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2026-03-30 15:00:00 | 965.80 | 2026-04-01 09:15:00 | 993.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-04-02 11:30:00 | 1013.00 | 2026-04-06 13:15:00 | 991.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-06 09:30:00 | 1013.25 | 2026-04-06 13:15:00 | 991.40 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest1 | 2026-04-13 09:15:00 | 955.45 | 2026-04-15 10:15:00 | 966.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1019.90 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2026-04-24 13:30:00 | 1019.30 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2026-04-24 14:30:00 | 1016.10 | 2026-04-24 15:15:00 | 1037.50 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest1 | 2026-04-30 12:30:00 | 1124.05 | 2026-05-05 09:15:00 | 1094.90 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2026-04-30 14:30:00 | 1130.45 | 2026-05-05 09:15:00 | 1094.90 | STOP_HIT | 1.00 | -3.14% |
