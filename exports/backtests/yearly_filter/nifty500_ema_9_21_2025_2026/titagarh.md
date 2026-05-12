# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 44 |
| ALERT2 | 43 |
| ALERT2_SKIP | 24 |
| ALERT3 | 100 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 32
- **Target hits / Stop hits / Partials:** 8 / 51 / 17
- **Avg / median % per leg:** 1.97% / 0.34%
- **Sum % (uncompounded):** 149.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 2 | 9.5% | 0 | 21 | 0 | -1.88% | -39.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 2 | 9.5% | 0 | 21 | 0 | -1.88% | -39.4% |
| SELL (all) | 55 | 42 | 76.4% | 8 | 30 | 17 | 3.44% | 189.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 42 | 76.4% | 8 | 30 | 17 | 3.44% | 189.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 76 | 44 | 57.9% | 8 | 51 | 17 | 1.97% | 149.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 740.00 | 705.32 | 703.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 741.35 | 717.71 | 709.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 935.65 | 935.89 | 896.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 894.00 | 935.89 | 896.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 895.55 | 927.82 | 896.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 895.80 | 927.82 | 896.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 904.00 | 923.06 | 897.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 920.40 | 910.03 | 899.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:00:00 | 916.30 | 913.61 | 904.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 892.45 | 903.35 | 903.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 892.45 | 903.35 | 903.80 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 906.05 | 902.84 | 902.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 919.25 | 909.63 | 906.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 923.30 | 925.66 | 918.48 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 908.25 | 916.34 | 916.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 895.55 | 909.83 | 912.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 909.40 | 899.58 | 904.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 909.40 | 899.58 | 904.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 904.10 | 900.48 | 904.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 898.00 | 900.48 | 904.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 901.10 | 901.07 | 904.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 13:15:00 | 912.90 | 903.39 | 904.96 | SL hit (close>static) qty=1.00 sl=909.80 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 929.60 | 904.48 | 903.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 934.85 | 910.55 | 906.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 930.70 | 932.14 | 925.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 930.70 | 932.14 | 925.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 941.70 | 949.67 | 944.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 941.45 | 949.67 | 944.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 942.45 | 948.22 | 944.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 942.45 | 948.22 | 944.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 942.00 | 946.75 | 944.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 936.95 | 946.75 | 944.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 946.15 | 946.63 | 944.64 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 929.45 | 942.20 | 943.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 906.25 | 928.09 | 935.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 897.90 | 897.22 | 906.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 897.90 | 897.22 | 906.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 865.50 | 852.30 | 863.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 865.50 | 852.30 | 863.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 863.85 | 854.61 | 863.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 859.00 | 854.61 | 863.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 860.00 | 857.35 | 863.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 883.30 | 865.48 | 865.87 | SL hit (close>static) qty=1.00 sl=874.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 885.10 | 869.41 | 867.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 888.50 | 873.23 | 869.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 936.05 | 943.04 | 933.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 936.05 | 943.04 | 933.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 937.00 | 941.84 | 933.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 951.00 | 941.84 | 933.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 943.60 | 947.96 | 948.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 943.60 | 947.96 | 948.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 941.25 | 946.62 | 947.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 944.00 | 938.76 | 942.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 944.00 | 938.76 | 942.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 941.50 | 939.30 | 942.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 942.95 | 939.30 | 942.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 939.05 | 939.25 | 941.80 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 949.00 | 943.41 | 943.23 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 938.10 | 943.37 | 943.62 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 946.50 | 943.46 | 943.42 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 936.00 | 943.76 | 944.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 929.25 | 935.79 | 939.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 934.55 | 934.50 | 937.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:45:00 | 934.70 | 934.50 | 937.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 938.65 | 935.33 | 937.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 938.65 | 935.33 | 937.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 937.90 | 935.85 | 937.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 944.50 | 935.85 | 937.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 940.50 | 936.78 | 937.97 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 942.85 | 938.84 | 938.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 946.40 | 940.35 | 939.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 946.05 | 949.55 | 946.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 946.05 | 949.55 | 946.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 944.15 | 948.47 | 946.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 947.10 | 947.72 | 946.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 936.60 | 945.06 | 945.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 936.60 | 945.06 | 945.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 930.70 | 939.47 | 942.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 938.20 | 933.07 | 936.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 940.00 | 933.07 | 936.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 935.60 | 933.58 | 936.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 934.25 | 935.76 | 936.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 887.54 | 900.19 | 910.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 896.75 | 890.47 | 901.20 | SL hit (close>ema200) qty=0.50 sl=890.47 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 862.20 | 856.42 | 856.16 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 853.10 | 855.76 | 855.88 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 858.15 | 856.23 | 856.09 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 841.35 | 853.41 | 854.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 836.00 | 842.88 | 847.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 11:15:00 | 843.20 | 842.94 | 847.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 12:00:00 | 843.20 | 842.94 | 847.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 854.00 | 842.86 | 845.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 854.00 | 842.86 | 845.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 856.15 | 845.51 | 846.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 845.05 | 845.51 | 846.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 14:15:00 | 802.80 | 825.90 | 835.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 802.10 | 790.92 | 807.05 | SL hit (close>ema200) qty=0.50 sl=790.92 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 813.05 | 807.11 | 806.98 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 804.00 | 806.95 | 807.33 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 808.60 | 807.52 | 807.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 820.45 | 810.66 | 808.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 818.50 | 819.06 | 815.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 819.10 | 819.06 | 815.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 859.55 | 860.97 | 855.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 849.50 | 860.97 | 855.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 852.60 | 863.19 | 859.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:30:00 | 852.00 | 863.19 | 859.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 851.75 | 860.90 | 859.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 851.75 | 860.90 | 859.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 847.85 | 857.35 | 857.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 13:15:00 | 845.20 | 854.92 | 856.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 837.00 | 835.31 | 842.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 839.10 | 835.31 | 842.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 842.60 | 837.87 | 841.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 843.00 | 837.87 | 841.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 842.70 | 838.83 | 841.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 841.00 | 838.83 | 841.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 850.00 | 842.72 | 842.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 851.95 | 842.72 | 842.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 853.00 | 844.78 | 843.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 859.85 | 847.79 | 845.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 850.05 | 850.23 | 847.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:45:00 | 849.50 | 850.23 | 847.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 848.60 | 849.91 | 847.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 852.95 | 848.43 | 847.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 844.55 | 847.65 | 847.37 | SL hit (close<static) qty=1.00 sl=847.10 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 844.75 | 847.07 | 847.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 841.70 | 846.00 | 846.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 842.50 | 840.91 | 842.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 843.25 | 840.91 | 842.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 839.95 | 840.72 | 842.64 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 849.00 | 844.05 | 843.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 857.00 | 848.22 | 846.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 903.85 | 904.11 | 891.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 903.85 | 904.11 | 891.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 931.20 | 939.04 | 929.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:30:00 | 930.75 | 939.04 | 929.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 934.65 | 936.90 | 930.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 940.85 | 936.90 | 930.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 15:00:00 | 937.70 | 937.06 | 930.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 937.15 | 944.46 | 944.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 937.15 | 944.46 | 944.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 922.35 | 935.76 | 940.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 911.75 | 909.27 | 919.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 889.35 | 903.88 | 911.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 890.25 | 900.94 | 909.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:00:00 | 892.10 | 889.46 | 898.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 893.40 | 890.80 | 898.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 882.00 | 877.91 | 881.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 882.00 | 877.91 | 881.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 886.00 | 879.53 | 881.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 888.25 | 879.53 | 881.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 883.00 | 881.12 | 882.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 880.60 | 881.12 | 882.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 886.00 | 882.09 | 882.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 886.00 | 882.09 | 882.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 889.35 | 883.54 | 883.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 889.35 | 883.54 | 883.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 890.95 | 886.17 | 884.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 882.95 | 885.53 | 884.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 885.95 | 885.53 | 884.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 879.45 | 884.31 | 883.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 879.30 | 884.31 | 883.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 885.80 | 884.61 | 884.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 887.40 | 884.89 | 884.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:15:00 | 886.90 | 885.17 | 884.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 889.90 | 900.54 | 901.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 889.90 | 900.54 | 901.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 882.50 | 888.03 | 892.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 891.30 | 887.80 | 891.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 891.30 | 887.80 | 891.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 890.70 | 888.38 | 891.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:45:00 | 888.05 | 889.85 | 891.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 888.40 | 889.50 | 890.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 886.20 | 880.85 | 880.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 887.00 | 882.08 | 881.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 887.00 | 882.08 | 881.46 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 878.00 | 881.14 | 881.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 876.30 | 880.34 | 880.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 879.25 | 878.47 | 879.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 879.25 | 878.47 | 879.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 878.10 | 878.39 | 879.67 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 890.15 | 880.75 | 880.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 894.70 | 886.36 | 883.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 892.60 | 895.07 | 890.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:45:00 | 893.85 | 895.07 | 890.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 896.45 | 895.35 | 890.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 889.95 | 895.35 | 890.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 891.15 | 894.32 | 891.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:15:00 | 900.00 | 894.32 | 891.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 894.75 | 897.88 | 898.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 894.75 | 897.88 | 898.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 883.50 | 895.01 | 896.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 912.60 | 896.92 | 897.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 912.60 | 896.92 | 897.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 908.00 | 899.14 | 898.34 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 889.05 | 898.04 | 899.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 883.50 | 895.13 | 897.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 849.80 | 848.04 | 858.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 849.80 | 848.04 | 858.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 855.40 | 849.61 | 854.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 855.40 | 849.61 | 854.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 849.75 | 849.63 | 854.00 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 867.60 | 856.52 | 856.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 886.95 | 863.31 | 859.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 872.60 | 876.19 | 869.78 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 863.70 | 867.39 | 867.85 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 871.80 | 868.33 | 868.20 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 861.10 | 866.88 | 867.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 893.80 | 872.07 | 869.62 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 869.70 | 873.05 | 873.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 15:15:00 | 869.10 | 872.26 | 872.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 877.10 | 870.64 | 871.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:45:00 | 875.10 | 870.64 | 871.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 878.40 | 872.19 | 872.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:30:00 | 872.80 | 872.37 | 872.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 873.65 | 872.63 | 872.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 873.65 | 872.63 | 872.58 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 869.60 | 872.02 | 872.30 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 874.30 | 872.45 | 872.41 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 866.25 | 871.28 | 871.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 856.50 | 867.56 | 870.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 857.45 | 856.45 | 862.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 857.45 | 856.45 | 862.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 846.65 | 842.56 | 847.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 846.20 | 842.56 | 847.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 844.05 | 842.86 | 847.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 844.05 | 842.86 | 847.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 847.20 | 843.73 | 847.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 847.20 | 843.73 | 847.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 844.15 | 843.81 | 847.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 843.00 | 843.81 | 847.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 848.05 | 845.12 | 847.30 | SL hit (close>static) qty=1.00 sl=847.60 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 772.45 | 768.73 | 768.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 774.00 | 770.47 | 769.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 788.65 | 789.05 | 783.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:30:00 | 789.20 | 789.05 | 783.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 784.95 | 788.60 | 784.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 784.75 | 788.60 | 784.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 782.20 | 787.32 | 784.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:15:00 | 780.70 | 787.32 | 784.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 781.60 | 786.18 | 784.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 781.60 | 786.18 | 784.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 775.95 | 782.07 | 782.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 774.45 | 780.54 | 781.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 776.50 | 776.09 | 779.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 776.50 | 776.09 | 779.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 770.70 | 773.21 | 776.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 774.70 | 773.21 | 776.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 777.90 | 772.07 | 774.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 777.90 | 772.07 | 774.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 785.15 | 774.69 | 775.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 785.15 | 774.69 | 775.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 784.05 | 776.56 | 776.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 801.55 | 781.56 | 778.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 850.75 | 851.23 | 835.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 873.05 | 851.23 | 835.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 890.25 | 900.69 | 886.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 922.70 | 902.81 | 888.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 908.05 | 903.73 | 891.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 878.20 | 897.10 | 890.65 | SL hit (close<static) qty=1.00 sl=886.45 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 882.50 | 890.30 | 890.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 881.60 | 888.56 | 889.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 887.40 | 885.75 | 887.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:15:00 | 889.60 | 885.75 | 887.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 889.60 | 886.52 | 887.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 886.20 | 886.70 | 887.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:45:00 | 885.90 | 886.64 | 887.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 891.80 | 888.22 | 888.29 | SL hit (close>static) qty=1.00 sl=891.60 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 823.80 | 807.63 | 806.02 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 801.00 | 806.65 | 807.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 789.85 | 801.58 | 804.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 782.50 | 774.38 | 780.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 793.80 | 774.38 | 780.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 786.80 | 776.86 | 781.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 781.60 | 779.32 | 781.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 782.90 | 779.28 | 781.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 796.60 | 783.53 | 783.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 796.60 | 783.53 | 783.15 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 777.35 | 782.87 | 783.59 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 792.00 | 783.60 | 783.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 807.75 | 790.55 | 786.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 804.65 | 804.93 | 796.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:45:00 | 802.45 | 804.93 | 796.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 814.05 | 806.83 | 801.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 817.05 | 808.87 | 802.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 816.90 | 810.43 | 804.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 820.65 | 811.90 | 805.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 817.25 | 813.67 | 807.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 813.50 | 820.54 | 813.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 813.50 | 820.54 | 813.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 798.75 | 816.18 | 811.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 800.80 | 816.18 | 811.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 792.95 | 811.54 | 810.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 792.95 | 811.54 | 810.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 785.40 | 806.31 | 807.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 785.40 | 806.31 | 807.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 776.25 | 794.66 | 801.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 791.95 | 787.28 | 795.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 791.95 | 787.28 | 795.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 794.30 | 788.68 | 795.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 799.50 | 788.68 | 795.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 795.50 | 790.05 | 795.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 790.25 | 796.32 | 796.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 792.55 | 796.56 | 796.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 794.00 | 794.62 | 795.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 788.45 | 783.60 | 783.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 788.45 | 783.60 | 783.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 794.90 | 787.46 | 785.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 785.10 | 790.89 | 788.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 785.20 | 790.89 | 788.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 780.65 | 788.84 | 787.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 779.20 | 788.84 | 787.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 778.65 | 785.38 | 786.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 770.70 | 779.88 | 783.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 764.25 | 758.55 | 765.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:30:00 | 768.20 | 758.55 | 765.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 757.15 | 758.18 | 762.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 757.15 | 758.18 | 762.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 763.20 | 759.18 | 763.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 763.20 | 759.18 | 763.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 763.00 | 759.95 | 763.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 768.10 | 759.95 | 763.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 761.65 | 760.29 | 762.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 759.85 | 761.95 | 762.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 721.86 | 730.10 | 736.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 683.87 | 700.19 | 710.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 686.80 | 669.25 | 667.42 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 651.40 | 667.83 | 668.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 640.80 | 655.47 | 658.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 662.15 | 656.30 | 658.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 662.15 | 656.30 | 658.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 660.75 | 657.19 | 658.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 656.00 | 657.40 | 658.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 651.20 | 657.82 | 658.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 623.20 | 639.14 | 647.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 618.64 | 639.14 | 647.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 629.95 | 627.80 | 637.54 | SL hit (close>ema200) qty=0.50 sl=627.80 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 647.70 | 635.30 | 635.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 650.00 | 643.84 | 639.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 635.05 | 642.08 | 639.50 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 630.85 | 637.67 | 637.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 622.70 | 634.68 | 636.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 636.10 | 631.55 | 634.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 637.15 | 631.55 | 634.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 634.65 | 632.17 | 634.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 635.55 | 632.17 | 634.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 632.00 | 632.13 | 634.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 629.40 | 632.13 | 634.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 629.20 | 631.45 | 633.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 628.65 | 630.56 | 632.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 597.93 | 620.94 | 627.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 597.74 | 620.94 | 627.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 597.22 | 612.62 | 622.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 606.00 | 604.43 | 613.88 | SL hit (close>ema200) qty=0.50 sl=604.43 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 637.60 | 618.44 | 617.48 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 602.75 | 620.24 | 621.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 600.70 | 611.18 | 616.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 613.10 | 588.73 | 596.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 617.00 | 588.73 | 596.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 608.40 | 592.66 | 597.89 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 617.05 | 601.14 | 601.06 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 595.20 | 600.71 | 601.30 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 611.80 | 603.11 | 602.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 614.45 | 605.38 | 603.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 707.55 | 708.62 | 693.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 707.10 | 708.62 | 693.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 690.80 | 707.67 | 703.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 699.45 | 705.74 | 702.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 699.25 | 705.74 | 702.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 700.85 | 702.20 | 701.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 697.10 | 701.18 | 701.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 697.10 | 701.18 | 701.36 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 718.90 | 704.06 | 702.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 11:15:00 | 733.95 | 719.28 | 712.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 738.20 | 739.45 | 731.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 747.15 | 740.92 | 738.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:45:00 | 746.50 | 740.97 | 738.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 727.25 | 736.64 | 737.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 727.25 | 736.64 | 737.66 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 755.10 | 738.66 | 737.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 15:15:00 | 760.00 | 742.92 | 739.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 789.75 | 790.27 | 780.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 789.75 | 790.27 | 780.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 786.30 | 788.42 | 781.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 773.00 | 788.42 | 781.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 763.50 | 783.43 | 779.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 763.50 | 783.43 | 779.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 764.00 | 779.55 | 778.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 759.00 | 779.55 | 778.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 766.75 | 776.99 | 777.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 761.95 | 768.98 | 771.58 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 796.00 | 774.38 | 773.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 863.85 | 792.28 | 781.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 835.30 | 836.87 | 821.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 835.30 | 836.87 | 821.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:45:00 | 920.40 | 2025-05-23 09:15:00 | 892.45 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-05-21 13:00:00 | 916.30 | 2025-05-23 09:15:00 | 892.45 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-06-02 11:15:00 | 898.00 | 2025-06-02 13:15:00 | 912.90 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-02 12:15:00 | 901.10 | 2025-06-02 13:15:00 | 912.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-03 10:45:00 | 899.90 | 2025-06-04 11:15:00 | 929.60 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-06-20 12:15:00 | 859.00 | 2025-06-23 09:15:00 | 883.30 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-06-20 14:00:00 | 860.00 | 2025-06-23 09:15:00 | 883.30 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-06-30 09:15:00 | 951.00 | 2025-07-03 12:15:00 | 943.60 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-17 15:00:00 | 947.10 | 2025-07-18 09:15:00 | 936.60 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-22 10:15:00 | 934.25 | 2025-07-25 11:15:00 | 887.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 934.25 | 2025-07-28 09:15:00 | 896.75 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-08-08 09:15:00 | 845.05 | 2025-08-08 14:15:00 | 802.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 845.05 | 2025-08-12 09:15:00 | 802.10 | STOP_HIT | 0.50 | 5.08% |
| BUY | retest2 | 2025-09-04 09:45:00 | 852.95 | 2025-09-04 10:15:00 | 844.55 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-16 14:15:00 | 940.85 | 2025-09-22 14:15:00 | 937.15 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-16 15:00:00 | 937.70 | 2025-09-22 14:15:00 | 937.15 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-09-26 09:15:00 | 889.35 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-09-26 11:15:00 | 890.25 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-09-29 10:00:00 | 892.10 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-09-29 10:30:00 | 893.40 | 2025-10-03 12:15:00 | 889.35 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-10-06 13:15:00 | 887.40 | 2025-10-13 09:15:00 | 889.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-10-06 14:15:00 | 886.90 | 2025-10-13 09:15:00 | 889.90 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-10-15 13:45:00 | 888.05 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-10-16 11:30:00 | 888.40 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-10-23 09:45:00 | 886.20 | 2025-10-23 10:15:00 | 887.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-29 10:15:00 | 900.00 | 2025-10-31 13:15:00 | 894.75 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-19 13:30:00 | 872.80 | 2025-11-19 14:15:00 | 873.65 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-11-26 13:15:00 | 843.00 | 2025-11-26 14:15:00 | 848.05 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-27 11:15:00 | 842.80 | 2025-12-03 10:15:00 | 800.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 840.00 | 2025-12-03 10:15:00 | 798.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:15:00 | 842.80 | 2025-12-04 15:15:00 | 791.40 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2025-12-01 09:30:00 | 840.00 | 2025-12-04 15:15:00 | 791.40 | STOP_HIT | 0.50 | 5.79% |
| BUY | retest2 | 2025-12-30 10:30:00 | 922.70 | 2025-12-30 14:15:00 | 878.20 | STOP_HIT | 1.00 | -4.82% |
| BUY | retest2 | 2025-12-30 12:30:00 | 908.05 | 2025-12-30 14:15:00 | 878.20 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-01-02 13:15:00 | 886.20 | 2026-01-02 15:15:00 | 891.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-02 13:45:00 | 885.90 | 2026-01-02 15:15:00 | 891.80 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-05 09:15:00 | 884.00 | 2026-01-08 10:15:00 | 839.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:15:00 | 886.00 | 2026-01-08 10:15:00 | 841.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 881.65 | 2026-01-08 10:15:00 | 837.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:00:00 | 882.85 | 2026-01-08 10:15:00 | 838.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 882.50 | 2026-01-08 10:15:00 | 838.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 882.45 | 2026-01-08 10:15:00 | 838.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 865.00 | 2026-01-08 15:15:00 | 821.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 884.00 | 2026-01-12 09:15:00 | 795.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 10:15:00 | 886.00 | 2026-01-12 09:15:00 | 797.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 881.65 | 2026-01-12 09:15:00 | 793.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:00:00 | 882.85 | 2026-01-12 09:15:00 | 794.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 882.50 | 2026-01-12 09:15:00 | 794.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-05 15:00:00 | 882.45 | 2026-01-12 09:15:00 | 794.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 865.00 | 2026-01-12 11:15:00 | 778.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 781.60 | 2026-01-22 14:15:00 | 796.60 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-01-22 12:30:00 | 782.90 | 2026-01-22 14:15:00 | 796.60 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-30 11:00:00 | 817.05 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-01-30 12:45:00 | 816.90 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2026-01-30 13:30:00 | 820.65 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-01-30 14:45:00 | 817.25 | 2026-02-01 14:15:00 | 785.40 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2026-02-04 09:15:00 | 790.25 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2026-02-04 10:15:00 | 792.55 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-02-04 15:00:00 | 794.00 | 2026-02-10 11:15:00 | 788.45 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-02-19 09:15:00 | 759.85 | 2026-02-25 13:15:00 | 721.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 759.85 | 2026-03-02 09:15:00 | 683.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 656.00 | 2026-03-16 09:15:00 | 623.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 651.20 | 2026-03-16 09:15:00 | 618.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 656.00 | 2026-03-16 14:15:00 | 629.95 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2026-03-13 09:15:00 | 651.20 | 2026-03-16 14:15:00 | 629.95 | STOP_HIT | 0.50 | 3.26% |
| SELL | retest2 | 2026-03-20 12:15:00 | 629.40 | 2026-03-23 10:15:00 | 597.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 629.20 | 2026-03-23 10:15:00 | 597.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 628.65 | 2026-03-23 12:15:00 | 597.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 629.40 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.72% |
| SELL | retest2 | 2026-03-20 13:30:00 | 629.20 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-03-20 14:30:00 | 628.65 | 2026-03-24 10:15:00 | 606.00 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2026-04-13 10:45:00 | 699.45 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-04-13 11:15:00 | 699.25 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-13 13:45:00 | 700.85 | 2026-04-13 14:15:00 | 697.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-04-23 09:15:00 | 747.15 | 2026-04-24 10:15:00 | 727.25 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-04-23 09:45:00 | 746.50 | 2026-04-24 10:15:00 | 727.25 | STOP_HIT | 1.00 | -2.58% |
