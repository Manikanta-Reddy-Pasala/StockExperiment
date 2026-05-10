# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 19
- **Target hits / Stop hits / Partials:** 3 / 23 / 8
- **Avg / median % per leg:** 0.10% / -0.55%
- **Sum % (uncompounded):** 3.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 15 | 44.1% | 3 | 23 | 8 | 0.10% | 3.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 15 | 44.1% | 3 | 23 | 8 | 0.10% | 3.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 15 | 44.1% | 3 | 23 | 8 | 0.10% | 3.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1005.00 | 1134.89 | 1135.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 1001.60 | 1130.04 | 1132.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1023.20 | 1022.39 | 1055.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 1023.20 | 1022.39 | 1055.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1092.00 | 1024.22 | 1055.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1092.60 | 1024.22 | 1055.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1073.20 | 1024.70 | 1055.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1066.80 | 1058.44 | 1067.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 1072.55 | 1058.58 | 1067.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 1071.75 | 1058.73 | 1067.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 1114.20 | 1059.39 | 1067.05 | SL hit (close>static) qty=1.00 sl=1092.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 1114.20 | 1059.39 | 1067.05 | SL hit (close>static) qty=1.00 sl=1092.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 1114.20 | 1059.39 | 1067.05 | SL hit (close>static) qty=1.00 sl=1092.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 1071.50 | 1061.01 | 1067.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1068.30 | 1060.98 | 1067.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1068.30 | 1060.98 | 1067.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1064.85 | 1061.01 | 1067.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1054.40 | 1061.01 | 1067.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 1072.40 | 1060.21 | 1066.74 | SL hit (close>static) qty=1.00 sl=1069.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 1056.80 | 1060.32 | 1066.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:30:00 | 1060.10 | 1060.35 | 1066.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 1056.40 | 1060.53 | 1066.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1061.60 | 1060.55 | 1066.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1061.60 | 1060.55 | 1066.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:15:00 | 1017.92 | 1059.28 | 1065.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:15:00 | 1007.09 | 1055.91 | 1063.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1003.96 | 1054.93 | 1063.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1003.58 | 1054.93 | 1063.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1041.90 | 1041.08 | 1054.24 | SL hit (close>ema200) qty=0.50 sl=1041.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1041.90 | 1041.08 | 1054.24 | SL hit (close>ema200) qty=0.50 sl=1041.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1041.90 | 1041.08 | 1054.24 | SL hit (close>ema200) qty=0.50 sl=1041.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 1041.90 | 1041.08 | 1054.24 | SL hit (close>ema200) qty=0.50 sl=1041.08 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1043.00 | 1041.24 | 1054.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 1033.90 | 1041.37 | 1053.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1028.80 | 1041.30 | 1053.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1075.00 | 1040.28 | 1052.64 | SL hit (close>static) qty=1.00 sl=1065.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 1075.00 | 1040.28 | 1052.64 | SL hit (close>static) qty=1.00 sl=1065.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:45:00 | 1036.20 | 1042.77 | 1053.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1035.00 | 1042.86 | 1053.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1047.80 | 1042.99 | 1053.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 1046.60 | 1042.99 | 1053.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 1062.00 | 1043.35 | 1053.15 | SL hit (close>static) qty=1.00 sl=1056.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1066.10 | 1043.74 | 1053.24 | SL hit (close>static) qty=1.00 sl=1065.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1066.10 | 1043.74 | 1053.24 | SL hit (close>static) qty=1.00 sl=1065.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1040.50 | 1045.10 | 1053.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 1045.80 | 1045.10 | 1053.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:00:00 | 1045.00 | 1045.10 | 1053.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1062.10 | 1045.26 | 1053.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 1062.10 | 1045.26 | 1053.56 | SL hit (close>static) qty=1.00 sl=1056.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 1062.10 | 1045.26 | 1053.56 | SL hit (close>static) qty=1.00 sl=1056.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 1062.10 | 1045.26 | 1053.56 | SL hit (close>static) qty=1.00 sl=1056.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 1062.10 | 1045.26 | 1053.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1077.50 | 1045.59 | 1053.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:45:00 | 1078.60 | 1045.59 | 1053.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 1059.50 | 1052.23 | 1056.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 1055.60 | 1052.23 | 1056.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1058.70 | 1052.29 | 1056.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1059.60 | 1052.29 | 1056.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1059.50 | 1052.36 | 1056.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1059.50 | 1052.36 | 1056.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1054.90 | 1052.39 | 1056.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1051.70 | 1052.39 | 1056.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1044.80 | 1052.31 | 1056.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 1041.00 | 1053.73 | 1056.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 1025.60 | 1053.32 | 1056.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1079.00 | 1052.55 | 1055.89 | SL hit (close>static) qty=1.00 sl=1063.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1079.00 | 1052.55 | 1055.89 | SL hit (close>static) qty=1.00 sl=1063.40 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-17 10:15:00 | 1084.50 | 1058.87 | 1058.86 | min_gap filter: gap=0.001% < 0.030% |
| TREND_RESET | 2025-11-17 10:15:00 | 1084.50 | 1058.87 | 1058.86 | EMA inversion without crossover edge (EMA200=1058.87 EMA400=1058.86) — end cycle |

### Cycle 2 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 984.00 | 1062.22 | 1062.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 978.10 | 1060.62 | 1061.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 911.50 | 897.87 | 946.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 911.30 | 897.87 | 946.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 924.15 | 898.36 | 945.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 907.85 | 899.78 | 944.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 862.46 | 898.59 | 942.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.30 | 894.28 | 935.66 | SL hit (close>ema200) qty=0.50 sl=894.28 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:00:00 | 913.05 | 922.61 | 943.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:30:00 | 902.75 | 921.05 | 941.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 984.25 | 920.42 | 940.79 | SL hit (close>static) qty=1.00 sl=946.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 984.25 | 920.42 | 940.79 | SL hit (close>static) qty=1.00 sl=946.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:30:00 | 910.00 | 920.69 | 940.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 940.00 | 918.41 | 936.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 938.40 | 918.41 | 936.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 930.95 | 918.53 | 936.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 919.50 | 918.88 | 935.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 15:15:00 | 915.00 | 917.75 | 934.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 864.50 | 915.70 | 931.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 873.52 | 915.70 | 931.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 869.25 | 915.70 | 931.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-05 13:15:00 | 827.55 | 904.90 | 924.83 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 819.00 | 898.37 | 920.51 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 823.50 | 898.37 | 920.51 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 15:15:00 | 922.00 | 863.58 | 879.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 922.80 | 864.76 | 879.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| CROSSOVER_SKIP | 2026-04-29 15:15:00 | 927.90 | 891.68 | 891.52 | min_gap filter: gap=0.018% < 0.030% |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 927.90 | 891.68 | 891.52 | Force close (TREND_INVERSION) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 927.90 | 891.68 | 891.52 | Force close (TREND_INVERSION) qty=1.00 alert=retest2 |
| TREND_RESET | 2026-04-29 15:15:00 | 927.90 | 891.68 | 891.52 | EMA inversion without crossover edge (EMA200=891.68 EMA400=891.52) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-26 09:15:00 | 1066.80 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-09-26 10:00:00 | 1072.55 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-09-26 11:15:00 | 1071.75 | 2025-09-29 14:15:00 | 1114.20 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1071.50 | 2025-10-03 14:15:00 | 1072.40 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-08 09:15:00 | 1017.92 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-09 10:15:00 | 1007.09 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-09 12:15:00 | 1003.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-09 12:15:00 | 1003.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1054.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1056.80 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2025-10-06 10:30:00 | 1060.10 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-10-07 10:00:00 | 1056.40 | 2025-10-16 13:15:00 | 1041.90 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-10-17 14:15:00 | 1033.90 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2025-10-20 09:15:00 | 1028.80 | 2025-10-23 09:15:00 | 1075.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-10-24 13:45:00 | 1036.20 | 2025-10-27 14:15:00 | 1062.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-27 09:30:00 | 1035.00 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-27 12:15:00 | 1046.60 | 2025-10-28 09:15:00 | 1066.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1040.50 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-10-29 11:15:00 | 1045.80 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-29 12:00:00 | 1045.00 | 2025-10-29 12:15:00 | 1062.10 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-11-10 15:15:00 | 1041.00 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-11-11 09:45:00 | 1025.60 | 2025-11-11 14:15:00 | 1079.00 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-01-29 14:15:00 | 862.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 907.85 | 2026-02-03 09:15:00 | 1001.30 | STOP_HIT | 0.50 | -10.29% |
| SELL | retest2 | 2026-02-11 14:00:00 | 913.05 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -7.80% |
| SELL | retest2 | 2026-02-12 14:30:00 | 902.75 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -9.03% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-02 09:15:00 | 864.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-02 09:15:00 | 873.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-02 09:15:00 | 869.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:30:00 | 910.00 | 2026-03-05 13:15:00 | 827.55 | TARGET_HIT | 0.50 | 9.06% |
| SELL | retest2 | 2026-02-23 13:45:00 | 919.50 | 2026-03-09 09:15:00 | 819.00 | TARGET_HIT | 0.50 | 10.93% |
| SELL | retest2 | 2026-02-25 15:15:00 | 915.00 | 2026-03-09 09:15:00 | 823.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-17 15:15:00 | 922.00 | 2026-04-29 15:15:00 | 927.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-04-20 09:30:00 | 922.80 | 2026-04-29 15:15:00 | 927.90 | STOP_HIT | 1.00 | -0.55% |
