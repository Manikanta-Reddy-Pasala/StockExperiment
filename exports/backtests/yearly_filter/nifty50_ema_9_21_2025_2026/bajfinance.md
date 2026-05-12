# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 954.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 66 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 22 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 35
- **Target hits / Stop hits / Partials:** 2 / 58 / 1
- **Avg / median % per leg:** -0.13% / -0.47%
- **Sum % (uncompounded):** -8.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 9 | 37.5% | 2 | 22 | 0 | 0.03% | 0.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 9 | 37.5% | 2 | 22 | 0 | 0.03% | 0.8% |
| SELL (all) | 37 | 17 | 45.9% | 0 | 36 | 1 | -0.24% | -9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 17 | 45.9% | 0 | 36 | 1 | -0.24% | -9.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 61 | 26 | 42.6% | 2 | 58 | 1 | -0.13% | -8.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 897.60 | 882.66 | 882.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 901.55 | 888.87 | 885.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 903.00 | 903.75 | 896.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 903.00 | 903.75 | 896.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 912.90 | 907.59 | 904.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:30:00 | 905.00 | 907.59 | 904.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 921.50 | 922.65 | 918.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 919.80 | 922.65 | 918.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 918.65 | 921.81 | 918.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 918.70 | 921.81 | 918.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 915.55 | 920.56 | 918.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 915.55 | 920.56 | 918.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 910.90 | 918.63 | 917.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 910.90 | 918.63 | 917.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 908.65 | 916.63 | 917.01 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 925.50 | 917.15 | 917.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 914.30 | 916.87 | 917.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 912.65 | 916.03 | 916.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 15:15:00 | 915.05 | 914.86 | 915.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 09:15:00 | 917.65 | 914.86 | 915.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 921.05 | 916.10 | 916.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 920.75 | 916.10 | 916.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 924.30 | 917.74 | 917.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 926.40 | 920.33 | 918.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 926.20 | 926.89 | 924.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 920.10 | 926.89 | 924.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 917.25 | 924.96 | 923.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 917.25 | 924.96 | 923.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 923.85 | 924.74 | 923.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 928.00 | 924.74 | 923.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 916.80 | 922.19 | 922.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 916.80 | 922.19 | 922.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 915.50 | 919.42 | 921.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 923.75 | 920.29 | 921.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 923.75 | 920.29 | 921.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 924.65 | 921.16 | 921.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:15:00 | 925.30 | 921.16 | 921.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 927.65 | 922.46 | 922.20 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 917.95 | 922.43 | 922.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 916.40 | 920.86 | 921.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 920.95 | 920.88 | 921.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 920.95 | 920.88 | 921.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 921.00 | 920.90 | 921.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 918.90 | 920.90 | 921.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 932.00 | 902.67 | 901.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 932.00 | 902.67 | 901.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 938.30 | 909.80 | 904.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 956.65 | 957.04 | 942.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 956.50 | 957.04 | 942.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 953.15 | 952.84 | 946.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 945.20 | 952.84 | 946.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 946.20 | 950.95 | 946.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 946.20 | 950.95 | 946.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 945.30 | 949.82 | 946.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 945.65 | 949.82 | 946.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 942.00 | 948.26 | 946.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 942.00 | 948.26 | 946.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 945.15 | 946.95 | 946.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 945.15 | 946.95 | 946.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 943.80 | 946.32 | 945.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 943.80 | 946.32 | 945.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 945.40 | 946.14 | 945.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 943.00 | 946.14 | 945.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 937.05 | 944.32 | 945.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 926.45 | 938.62 | 942.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 932.05 | 931.69 | 937.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 933.15 | 931.69 | 937.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 934.00 | 932.20 | 936.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 939.50 | 932.20 | 936.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 931.50 | 932.06 | 935.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 923.00 | 932.59 | 935.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 921.00 | 930.17 | 933.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 923.50 | 926.05 | 929.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 924.50 | 925.84 | 929.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 904.00 | 901.91 | 906.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 904.00 | 901.91 | 906.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 912.00 | 903.93 | 907.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 909.50 | 903.93 | 907.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 910.00 | 905.14 | 907.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 912.00 | 905.14 | 907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 915.50 | 907.21 | 908.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 915.50 | 907.21 | 908.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 913.00 | 909.78 | 909.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 913.00 | 909.78 | 909.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 924.50 | 912.72 | 910.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 920.00 | 921.01 | 916.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 920.00 | 921.01 | 916.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 916.00 | 920.00 | 916.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 916.00 | 920.00 | 916.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 919.00 | 919.80 | 916.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 922.00 | 919.80 | 916.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 921.50 | 921.26 | 918.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 933.50 | 937.16 | 937.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 933.50 | 937.16 | 937.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 926.60 | 934.64 | 936.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 933.55 | 918.96 | 922.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 937.25 | 918.96 | 922.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 932.50 | 921.67 | 923.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 927.00 | 922.29 | 923.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 924.80 | 923.90 | 923.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 924.80 | 923.90 | 923.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 927.60 | 924.92 | 924.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 941.45 | 944.18 | 939.40 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 932.40 | 937.21 | 937.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 919.75 | 933.72 | 935.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 923.15 | 922.40 | 927.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 923.15 | 922.40 | 927.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 926.95 | 922.99 | 926.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 927.75 | 922.99 | 926.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 928.65 | 924.12 | 926.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 928.65 | 924.12 | 926.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 929.00 | 925.10 | 926.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 921.25 | 925.10 | 926.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 935.50 | 924.51 | 924.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 935.50 | 924.51 | 924.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 13:15:00 | 936.80 | 927.44 | 925.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 942.50 | 944.20 | 938.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:30:00 | 941.30 | 944.20 | 938.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 956.70 | 961.15 | 955.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 956.70 | 961.15 | 955.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 958.85 | 960.69 | 956.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:30:00 | 959.70 | 960.65 | 956.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 960.50 | 960.65 | 956.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 960.35 | 960.37 | 956.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 918.10 | 951.28 | 953.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 918.10 | 951.28 | 953.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 891.55 | 911.35 | 926.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 889.05 | 889.03 | 903.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 891.40 | 889.03 | 903.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 885.00 | 882.10 | 887.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 885.00 | 882.10 | 887.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 889.70 | 883.16 | 886.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 889.70 | 883.16 | 886.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 890.20 | 884.57 | 886.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 890.20 | 884.57 | 886.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 881.45 | 883.94 | 886.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 880.35 | 883.41 | 885.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:00:00 | 873.90 | 881.51 | 884.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 879.10 | 880.29 | 883.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 880.95 | 880.85 | 883.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 885.00 | 881.68 | 883.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 885.00 | 881.68 | 883.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 882.05 | 881.75 | 883.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 888.65 | 884.44 | 884.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 888.65 | 884.44 | 884.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 892.85 | 887.27 | 885.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 882.60 | 887.14 | 886.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 882.60 | 887.14 | 886.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 875.75 | 884.86 | 885.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 872.05 | 878.63 | 881.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 877.25 | 875.78 | 879.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 877.25 | 875.78 | 879.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 880.05 | 876.63 | 879.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 880.05 | 876.63 | 879.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 881.90 | 877.69 | 879.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 885.45 | 877.69 | 879.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 882.25 | 878.60 | 879.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 876.40 | 878.61 | 879.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 876.85 | 877.69 | 878.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 873.55 | 877.35 | 878.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 915.20 | 871.06 | 866.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 915.20 | 871.06 | 866.59 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 886.65 | 895.22 | 896.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 881.05 | 891.03 | 893.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 885.05 | 881.28 | 884.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 885.05 | 881.28 | 884.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 887.60 | 882.54 | 885.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 887.60 | 882.54 | 885.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 880.95 | 882.22 | 884.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 878.80 | 881.74 | 884.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 876.50 | 880.69 | 883.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 888.60 | 882.00 | 883.66 | SL hit (close>static) qty=1.00 sl=887.95 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 889.30 | 885.32 | 884.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 898.30 | 889.25 | 886.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 888.35 | 891.15 | 888.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 888.35 | 891.15 | 888.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 890.40 | 891.00 | 888.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 891.70 | 891.03 | 889.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 891.60 | 891.14 | 889.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-12 09:15:00 | 980.87 | 971.97 | 963.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 994.75 | 1001.61 | 1002.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 994.00 | 1000.09 | 1001.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 995.05 | 994.71 | 997.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 14:00:00 | 995.05 | 994.71 | 997.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1000.25 | 995.44 | 997.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1002.40 | 995.44 | 997.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1006.20 | 997.59 | 997.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1008.50 | 997.59 | 997.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 1000.90 | 998.25 | 998.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 1021.35 | 1006.20 | 1002.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1027.10 | 1027.26 | 1021.04 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1012.50 | 1019.22 | 1019.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1002.10 | 1015.80 | 1017.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 992.25 | 991.41 | 999.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 992.25 | 991.41 | 999.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 996.00 | 992.55 | 998.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 995.90 | 992.55 | 998.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 999.80 | 994.00 | 998.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 999.80 | 994.00 | 998.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 994.80 | 994.16 | 998.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 989.10 | 996.73 | 998.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1013.95 | 992.42 | 991.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1013.95 | 992.42 | 991.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1024.05 | 1009.35 | 1001.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1014.95 | 1015.69 | 1008.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1014.95 | 1015.69 | 1008.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1017.85 | 1020.92 | 1016.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1017.85 | 1020.92 | 1016.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1024.55 | 1023.82 | 1020.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1026.35 | 1024.32 | 1020.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1026.80 | 1024.48 | 1021.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1026.20 | 1024.57 | 1022.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1026.65 | 1025.89 | 1023.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1026.70 | 1030.44 | 1027.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1026.70 | 1030.44 | 1027.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1020.45 | 1028.44 | 1026.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1020.45 | 1028.44 | 1026.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1018.00 | 1026.35 | 1025.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 1018.25 | 1026.35 | 1025.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 1015.60 | 1024.20 | 1024.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1015.60 | 1024.20 | 1024.80 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1041.00 | 1026.51 | 1025.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 1046.75 | 1030.56 | 1027.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 13:15:00 | 1089.05 | 1091.03 | 1085.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 13:45:00 | 1088.90 | 1091.03 | 1085.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1081.75 | 1088.56 | 1085.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 1081.95 | 1088.56 | 1085.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1079.80 | 1086.81 | 1085.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1079.40 | 1086.81 | 1085.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1076.10 | 1083.55 | 1084.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 1072.15 | 1081.02 | 1082.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1075.25 | 1074.56 | 1078.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 1075.25 | 1074.56 | 1078.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1067.30 | 1073.33 | 1077.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:00:00 | 1064.65 | 1071.60 | 1076.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 1064.60 | 1068.94 | 1073.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 1063.90 | 1067.09 | 1072.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 1065.35 | 1056.75 | 1062.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1055.70 | 1056.54 | 1061.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 1055.20 | 1055.32 | 1060.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:00:00 | 1055.30 | 1048.56 | 1049.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 15:15:00 | 1055.00 | 1050.23 | 1050.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1055.00 | 1051.18 | 1050.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 1055.00 | 1051.18 | 1050.68 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1043.00 | 1049.55 | 1049.98 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 1062.50 | 1048.74 | 1048.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 1072.00 | 1053.39 | 1050.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1007.70 | 1063.15 | 1062.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1011.90 | 1063.15 | 1062.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1002.70 | 1051.06 | 1056.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1001.10 | 1041.07 | 1051.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1010.00 | 1009.09 | 1015.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 1010.00 | 1009.09 | 1015.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1014.30 | 1010.57 | 1014.93 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1028.10 | 1018.38 | 1017.55 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1013.30 | 1019.14 | 1019.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1013.10 | 1016.17 | 1017.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1012.70 | 1010.16 | 1013.31 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1029.40 | 1017.66 | 1016.13 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 1009.20 | 1015.85 | 1016.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1003.00 | 1012.26 | 1014.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 999.70 | 993.12 | 998.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 999.70 | 993.12 | 998.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1002.60 | 995.02 | 998.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1002.60 | 995.02 | 998.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1011.00 | 998.21 | 999.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1011.00 | 998.21 | 999.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1011.90 | 1002.71 | 1001.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 1033.60 | 1011.35 | 1006.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 1033.60 | 1034.72 | 1026.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 1033.60 | 1034.72 | 1026.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1030.20 | 1034.66 | 1028.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1030.20 | 1034.66 | 1028.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1030.00 | 1033.73 | 1028.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 1030.70 | 1033.73 | 1028.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1024.10 | 1031.80 | 1028.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1024.10 | 1031.80 | 1028.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1024.00 | 1030.24 | 1027.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1022.20 | 1030.24 | 1027.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 1019.80 | 1025.84 | 1026.26 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 1032.50 | 1026.83 | 1026.62 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1020.00 | 1025.47 | 1026.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1018.00 | 1023.02 | 1024.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1025.90 | 1023.59 | 1024.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1025.90 | 1023.59 | 1024.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1025.30 | 1023.94 | 1024.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1023.10 | 1023.94 | 1024.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1021.90 | 1023.53 | 1024.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1017.40 | 1023.53 | 1024.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:30:00 | 1016.20 | 1021.65 | 1023.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1026.40 | 1023.75 | 1023.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1026.40 | 1023.75 | 1023.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 14:15:00 | 1029.50 | 1025.49 | 1024.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1025.80 | 1041.30 | 1036.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1025.80 | 1041.30 | 1036.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1030.40 | 1039.12 | 1035.69 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1026.10 | 1033.89 | 1033.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1017.20 | 1028.51 | 1031.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1014.20 | 1013.44 | 1018.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:15:00 | 1016.60 | 1013.44 | 1018.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1015.70 | 1013.90 | 1018.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1019.10 | 1013.90 | 1018.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1013.30 | 1013.78 | 1017.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1010.70 | 1013.78 | 1017.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1016.40 | 1010.38 | 1014.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1016.60 | 1010.38 | 1014.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1018.40 | 1011.98 | 1014.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1018.40 | 1011.98 | 1014.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1017.90 | 1014.45 | 1015.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 1017.90 | 1014.45 | 1015.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1017.90 | 1015.14 | 1015.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1012.70 | 1015.27 | 1015.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 1005.50 | 1003.98 | 1003.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1005.50 | 1003.98 | 1003.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1008.90 | 1004.96 | 1004.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1008.10 | 1009.25 | 1007.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1007.70 | 1009.25 | 1007.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1011.20 | 1009.64 | 1007.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1012.90 | 1010.51 | 1008.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:00:00 | 1012.50 | 1010.91 | 1008.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1022.90 | 1011.13 | 1009.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1002.70 | 1014.43 | 1013.81 | SL hit (close<static) qty=1.00 sl=1007.40 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1005.50 | 1012.64 | 1013.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 1000.30 | 1008.90 | 1011.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 998.70 | 997.15 | 1002.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 998.70 | 997.15 | 1002.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 986.15 | 979.90 | 984.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 987.45 | 979.90 | 984.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 984.65 | 980.85 | 984.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 988.90 | 980.85 | 984.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 990.00 | 984.57 | 985.42 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 990.55 | 986.76 | 986.33 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 982.80 | 985.82 | 986.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 12:15:00 | 978.80 | 983.89 | 985.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 983.00 | 981.44 | 983.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:45:00 | 982.95 | 981.44 | 983.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 978.00 | 980.75 | 982.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:45:00 | 977.10 | 980.26 | 982.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:00:00 | 973.95 | 978.67 | 981.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 976.95 | 973.51 | 973.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 956.60 | 952.06 | 951.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 956.60 | 952.06 | 951.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 10:15:00 | 966.80 | 955.01 | 953.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 945.05 | 960.55 | 957.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 943.80 | 960.55 | 957.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 944.00 | 957.24 | 956.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 944.00 | 957.24 | 956.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 942.80 | 954.35 | 955.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 933.30 | 947.25 | 951.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 940.00 | 939.31 | 945.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 940.00 | 939.31 | 945.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 943.10 | 940.07 | 945.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:30:00 | 944.75 | 940.07 | 945.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 942.15 | 939.23 | 943.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 935.20 | 939.60 | 941.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 935.60 | 929.07 | 929.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 935.60 | 929.07 | 929.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 15:15:00 | 937.25 | 933.29 | 931.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 929.10 | 932.45 | 931.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 928.80 | 932.45 | 931.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 927.90 | 931.54 | 930.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 927.90 | 931.54 | 930.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 936.55 | 933.48 | 932.03 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 924.70 | 931.07 | 931.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 909.80 | 926.81 | 929.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 908.00 | 905.27 | 912.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 959.85 | 905.27 | 912.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 946.65 | 913.54 | 915.85 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 953.10 | 921.45 | 919.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 961.20 | 934.92 | 926.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 957.30 | 961.53 | 951.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 957.30 | 961.53 | 951.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 964.50 | 976.80 | 972.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 964.50 | 976.80 | 972.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 964.70 | 974.38 | 972.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 970.10 | 971.76 | 971.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:00:00 | 968.85 | 971.18 | 971.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 14:15:00 | 965.50 | 970.04 | 970.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 965.50 | 970.04 | 970.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 963.95 | 968.04 | 969.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 970.50 | 968.53 | 969.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 972.00 | 968.53 | 969.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 967.90 | 968.41 | 969.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 970.10 | 968.41 | 969.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 969.65 | 968.65 | 969.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 969.65 | 968.65 | 969.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 970.75 | 969.07 | 969.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 967.10 | 968.81 | 969.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 967.05 | 968.81 | 969.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 977.95 | 970.36 | 969.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 977.95 | 970.36 | 969.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 10:15:00 | 987.80 | 973.84 | 971.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 1015.10 | 1016.13 | 1003.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 1015.10 | 1016.13 | 1003.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1012.70 | 1013.02 | 1005.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1017.85 | 1011.03 | 1008.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 1016.05 | 1016.00 | 1015.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1016.40 | 1028.78 | 1026.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 1020.20 | 1025.13 | 1025.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 1020.20 | 1025.13 | 1025.32 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1028.05 | 1025.39 | 1025.25 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 11:15:00 | 1024.20 | 1025.15 | 1025.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 12:15:00 | 1014.80 | 1023.08 | 1024.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 1022.30 | 1021.75 | 1023.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 1022.30 | 1021.75 | 1023.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1022.35 | 1021.65 | 1023.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 1019.65 | 1021.65 | 1023.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1023.50 | 1022.02 | 1023.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 1023.50 | 1022.02 | 1023.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1023.60 | 1022.34 | 1023.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1015.10 | 1020.11 | 1022.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 964.35 | 975.83 | 990.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 957.15 | 956.16 | 971.23 | SL hit (close>ema200) qty=0.50 sl=956.16 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 880.35 | 871.19 | 871.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 882.95 | 873.54 | 872.15 | Break + close above crossover candle high |

### Cycle 58 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 847.50 | 868.33 | 869.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 844.10 | 863.49 | 867.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 815.05 | 812.31 | 826.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:00:00 | 815.05 | 812.31 | 826.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 822.50 | 814.39 | 824.08 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 849.90 | 832.22 | 829.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 852.00 | 836.18 | 831.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 854.15 | 868.88 | 855.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 854.15 | 868.88 | 855.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 855.65 | 866.23 | 855.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:15:00 | 850.70 | 866.23 | 855.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 850.20 | 863.03 | 855.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 850.20 | 863.03 | 855.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 848.60 | 860.14 | 854.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:45:00 | 847.70 | 860.14 | 854.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 827.00 | 847.73 | 850.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 818.40 | 841.86 | 847.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 832.80 | 820.87 | 832.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 825.70 | 827.88 | 832.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 825.10 | 822.25 | 823.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 843.00 | 827.78 | 826.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 843.00 | 827.78 | 826.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 847.30 | 831.68 | 828.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 893.90 | 903.42 | 888.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 893.90 | 903.42 | 888.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 902.40 | 914.88 | 905.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 919.65 | 903.95 | 903.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 908.20 | 910.91 | 910.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 907.10 | 910.15 | 910.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 907.10 | 910.15 | 910.51 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 11:15:00 | 916.20 | 910.67 | 910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 14:15:00 | 919.30 | 914.03 | 912.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 932.20 | 933.22 | 925.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 10:00:00 | 932.20 | 933.22 | 925.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 919.85 | 932.24 | 929.18 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 911.15 | 924.89 | 926.18 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 926.90 | 921.34 | 921.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 930.40 | 924.75 | 923.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 935.45 | 940.08 | 934.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 935.45 | 940.08 | 934.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 939.70 | 940.00 | 934.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 946.00 | 940.00 | 934.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 941.55 | 946.68 | 942.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 953.50 | 962.53 | 963.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 953.50 | 962.53 | 963.41 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 11:15:00 | 928.00 | 2025-05-27 13:15:00 | 916.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-05-30 09:15:00 | 918.90 | 2025-06-06 10:15:00 | 932.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-06-17 11:00:00 | 923.00 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2025-06-17 11:45:00 | 921.00 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-18 10:45:00 | 923.50 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-06-18 12:15:00 | 924.50 | 2025-06-23 15:15:00 | 913.00 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-06-25 09:15:00 | 922.00 | 2025-07-01 10:15:00 | 933.50 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-06-25 12:45:00 | 921.50 | 2025-07-01 10:15:00 | 933.50 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-07-04 11:30:00 | 927.00 | 2025-07-07 13:15:00 | 924.80 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-16 09:15:00 | 921.25 | 2025-07-18 09:15:00 | 935.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-07-24 13:30:00 | 959.70 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-07-24 14:00:00 | 960.50 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-07-24 14:30:00 | 960.35 | 2025-07-25 09:15:00 | 918.10 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-08-01 14:15:00 | 880.35 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-01 15:00:00 | 873.90 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-04 09:45:00 | 879.10 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-08-04 10:45:00 | 880.95 | 2025-08-05 09:15:00 | 888.65 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-08 15:00:00 | 876.40 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-08-11 11:30:00 | 876.85 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-08-12 09:15:00 | 873.55 | 2025-08-18 09:15:00 | 915.20 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2025-08-29 13:45:00 | 878.80 | 2025-09-01 09:15:00 | 888.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-08-29 15:00:00 | 876.50 | 2025-09-01 09:15:00 | 888.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-03 11:45:00 | 891.70 | 2025-09-12 09:15:00 | 980.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-03 13:00:00 | 891.60 | 2025-09-12 09:15:00 | 980.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 989.10 | 2025-10-06 09:15:00 | 1013.95 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1026.35 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1026.80 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-13 09:15:00 | 1026.20 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-13 10:45:00 | 1026.65 | 2025-10-14 12:15:00 | 1015.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-29 11:00:00 | 1064.65 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-10-29 13:15:00 | 1064.60 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-10-29 14:30:00 | 1063.90 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-10-31 10:00:00 | 1065.35 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-10-31 11:45:00 | 1055.20 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-11-04 14:00:00 | 1055.30 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-11-04 15:15:00 | 1055.00 | 2025-11-04 15:15:00 | 1055.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1017.40 | 2025-12-04 12:15:00 | 1026.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-03 11:30:00 | 1016.20 | 2025-12-04 12:15:00 | 1026.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1012.70 | 2025-12-19 13:15:00 | 1005.50 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1012.90 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-23 15:00:00 | 1012.50 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1022.90 | 2025-12-26 09:15:00 | 1002.70 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-06 11:45:00 | 977.10 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2026-01-06 14:00:00 | 973.95 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2026-01-09 10:45:00 | 976.95 | 2026-01-19 09:15:00 | 956.60 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2026-01-23 11:30:00 | 935.20 | 2026-01-28 15:15:00 | 935.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-10 12:45:00 | 970.10 | 2026-02-10 14:15:00 | 965.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-10 14:00:00 | 968.85 | 2026-02-10 14:15:00 | 965.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-11 14:30:00 | 967.10 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-11 15:15:00 | 967.05 | 2026-02-12 09:15:00 | 977.95 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1017.85 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2026-02-20 09:30:00 | 1016.05 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1016.40 | 2026-02-24 12:15:00 | 1020.20 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1015.10 | 2026-03-04 09:15:00 | 964.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1015.10 | 2026-03-05 09:15:00 | 957.15 | STOP_HIT | 0.50 | 5.71% |
| SELL | retest2 | 2026-04-01 13:30:00 | 825.70 | 2026-04-06 12:15:00 | 843.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-06 11:00:00 | 825.10 | 2026-04-06 12:15:00 | 843.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-04-15 09:15:00 | 919.65 | 2026-04-17 10:15:00 | 907.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-04-17 09:45:00 | 908.20 | 2026-04-17 10:15:00 | 907.10 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-05-04 09:15:00 | 946.00 | 2026-05-08 12:15:00 | 953.50 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2026-05-05 10:15:00 | 941.55 | 2026-05-08 12:15:00 | 953.50 | STOP_HIT | 1.00 | 1.27% |
