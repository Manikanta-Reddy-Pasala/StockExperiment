# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 808.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 161 |
| ALERT1 | 102 |
| ALERT2 | 101 |
| ALERT2_SKIP | 57 |
| ALERT3 | 264 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 130 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 103
- **Target hits / Stop hits / Partials:** 13 / 121 / 16
- **Avg / median % per leg:** 0.40% / -0.93%
- **Sum % (uncompounded):** 60.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 14 | 26.9% | 8 | 44 | 0 | 0.67% | 34.6% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.83% | -5.5% |
| BUY @ 3rd Alert (retest2) | 49 | 14 | 28.6% | 8 | 41 | 0 | 0.82% | 40.1% |
| SELL (all) | 98 | 33 | 33.7% | 5 | 77 | 16 | 0.26% | 25.7% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.17% | 8.3% |
| SELL @ 3rd Alert (retest2) | 96 | 31 | 32.3% | 5 | 76 | 15 | 0.18% | 17.4% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.57% | 2.8% |
| retest2 (combined) | 145 | 45 | 31.0% | 13 | 117 | 15 | 0.40% | 57.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 898.95 | 888.30 | 887.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 901.10 | 890.86 | 888.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 891.40 | 897.01 | 893.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 891.40 | 897.01 | 893.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 891.40 | 897.01 | 893.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 891.40 | 897.01 | 893.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 895.05 | 896.62 | 893.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 899.55 | 896.25 | 893.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 11:45:00 | 899.00 | 896.64 | 894.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 899.35 | 896.53 | 894.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:30:00 | 899.05 | 898.66 | 897.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 900.65 | 899.28 | 897.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 897.25 | 898.34 | 898.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 897.25 | 898.34 | 898.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 887.75 | 893.37 | 895.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 894.75 | 891.47 | 892.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 894.75 | 891.47 | 892.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 894.75 | 891.47 | 892.99 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 918.20 | 898.01 | 895.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 15:15:00 | 930.00 | 904.41 | 898.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 13:15:00 | 906.25 | 906.28 | 902.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:45:00 | 906.00 | 906.28 | 902.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 906.00 | 906.18 | 902.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 880.70 | 906.18 | 902.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 892.60 | 903.46 | 901.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 884.60 | 903.46 | 901.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 882.55 | 899.28 | 900.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 854.15 | 890.25 | 895.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 14:15:00 | 883.00 | 880.89 | 889.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 15:00:00 | 883.00 | 880.89 | 889.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 895.00 | 883.71 | 890.01 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 906.55 | 893.66 | 893.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 947.20 | 908.60 | 900.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 14:15:00 | 915.95 | 918.95 | 909.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:30:00 | 919.75 | 918.95 | 909.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1054.15 | 1064.76 | 1055.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1054.15 | 1064.76 | 1055.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1051.55 | 1062.12 | 1054.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 1051.95 | 1062.12 | 1054.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1055.30 | 1059.61 | 1054.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:15:00 | 1058.75 | 1057.37 | 1054.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:45:00 | 1060.30 | 1057.71 | 1054.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 10:30:00 | 1060.55 | 1058.28 | 1055.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 1048.50 | 1056.57 | 1055.20 | SL hit (close<static) qty=1.00 sl=1052.55 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 1045.50 | 1053.26 | 1054.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 10:15:00 | 1038.00 | 1050.21 | 1052.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 1048.00 | 1040.20 | 1045.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 1048.00 | 1040.20 | 1045.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1048.00 | 1040.20 | 1045.18 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1060.85 | 1047.96 | 1046.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 12:15:00 | 1066.65 | 1051.70 | 1048.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 1066.60 | 1073.36 | 1066.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:00:00 | 1066.60 | 1073.36 | 1066.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1062.70 | 1071.23 | 1066.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 1062.70 | 1071.23 | 1066.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1052.35 | 1067.45 | 1064.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 1052.35 | 1067.45 | 1064.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1050.00 | 1061.81 | 1062.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 11:15:00 | 1048.50 | 1055.24 | 1058.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 1052.25 | 1044.51 | 1051.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 1052.25 | 1044.51 | 1051.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1052.25 | 1044.51 | 1051.18 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 1069.50 | 1055.10 | 1054.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1128.95 | 1075.92 | 1065.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 13:15:00 | 1089.10 | 1089.44 | 1076.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 13:45:00 | 1088.80 | 1089.44 | 1076.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1084.55 | 1088.23 | 1083.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 1094.75 | 1088.02 | 1085.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:30:00 | 1098.35 | 1087.96 | 1085.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 13:30:00 | 1096.70 | 1088.23 | 1086.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 1096.80 | 1089.82 | 1087.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1086.30 | 1090.59 | 1088.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 1087.95 | 1090.59 | 1088.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1085.00 | 1089.47 | 1087.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 1085.00 | 1089.47 | 1087.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1086.15 | 1088.81 | 1087.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 1083.35 | 1086.41 | 1086.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 1083.35 | 1086.41 | 1086.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1072.50 | 1080.26 | 1082.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 1098.35 | 1071.89 | 1072.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 1098.35 | 1071.89 | 1072.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1098.35 | 1071.89 | 1072.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 1098.35 | 1071.89 | 1072.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 1109.00 | 1079.31 | 1075.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1115.40 | 1093.50 | 1087.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 1089.45 | 1095.97 | 1090.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 1089.45 | 1095.97 | 1090.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1089.45 | 1095.97 | 1090.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1089.45 | 1095.97 | 1090.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1095.20 | 1095.82 | 1091.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 1094.40 | 1095.82 | 1091.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1095.00 | 1095.65 | 1091.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 1093.65 | 1095.65 | 1091.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1085.85 | 1093.69 | 1091.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1086.00 | 1093.69 | 1091.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1080.95 | 1091.14 | 1090.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1080.95 | 1091.14 | 1090.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 11:15:00 | 1081.80 | 1089.28 | 1089.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 14:15:00 | 1066.00 | 1082.39 | 1086.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 09:15:00 | 1100.05 | 1085.06 | 1086.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 1100.05 | 1085.06 | 1086.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1100.05 | 1085.06 | 1086.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 1110.25 | 1085.06 | 1086.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1090.05 | 1086.06 | 1086.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 1100.10 | 1086.06 | 1086.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 11:15:00 | 1099.00 | 1088.65 | 1087.97 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 1073.35 | 1086.03 | 1087.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 15:15:00 | 1065.95 | 1082.01 | 1085.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 11:15:00 | 995.45 | 995.07 | 1006.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:00:00 | 995.45 | 995.07 | 1006.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 999.95 | 996.77 | 1003.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:15:00 | 991.00 | 996.77 | 1003.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:00:00 | 993.30 | 997.25 | 1002.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 981.20 | 999.56 | 1002.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 993.55 | 994.14 | 997.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 986.80 | 992.68 | 996.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 983.50 | 989.09 | 992.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 10:00:00 | 982.45 | 987.76 | 991.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:45:00 | 983.40 | 987.91 | 990.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 943.63 | 961.19 | 971.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 943.87 | 961.19 | 971.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 941.45 | 945.03 | 957.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 932.14 | 945.03 | 957.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 934.32 | 945.03 | 957.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 933.33 | 945.03 | 957.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 934.23 | 945.03 | 957.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-05 15:15:00 | 891.90 | 915.62 | 935.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 943.00 | 931.58 | 931.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 945.00 | 936.09 | 933.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 935.65 | 939.74 | 936.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 14:15:00 | 935.65 | 939.74 | 936.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 935.65 | 939.74 | 936.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 935.65 | 939.74 | 936.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 937.00 | 939.19 | 936.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 940.70 | 939.19 | 936.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:00:00 | 939.15 | 937.08 | 936.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 14:30:00 | 940.00 | 936.86 | 936.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 15:15:00 | 939.00 | 936.86 | 936.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 939.00 | 937.29 | 936.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 928.40 | 937.29 | 936.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 928.00 | 935.43 | 935.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 928.00 | 935.43 | 935.95 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 11:15:00 | 940.00 | 934.83 | 934.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 955.75 | 943.25 | 939.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 992.00 | 992.41 | 980.74 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1000.85 | 992.41 | 980.74 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 998.90 | 993.24 | 982.18 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 985.00 | 990.22 | 984.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 983.10 | 990.22 | 984.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 982.70 | 988.72 | 984.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-21 15:15:00 | 982.70 | 988.72 | 984.55 | SL hit (close<ema400) qty=1.00 sl=984.55 alert=retest1 |

### Cycle 18 — SELL (started 2024-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 09:15:00 | 980.60 | 983.48 | 983.54 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 985.35 | 983.86 | 983.71 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 981.00 | 983.26 | 983.50 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 985.75 | 983.88 | 983.75 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 11:15:00 | 980.75 | 983.17 | 983.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 12:15:00 | 978.70 | 981.93 | 982.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 982.00 | 978.97 | 980.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 982.00 | 978.97 | 980.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 982.00 | 978.97 | 980.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:15:00 | 988.95 | 978.97 | 980.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 1004.35 | 984.05 | 982.85 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 985.00 | 988.56 | 988.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 982.95 | 986.72 | 987.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 986.75 | 986.25 | 987.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 986.75 | 986.25 | 987.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 986.75 | 986.25 | 987.22 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 989.45 | 987.68 | 987.67 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 987.35 | 987.61 | 987.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 983.10 | 986.71 | 987.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 987.45 | 986.72 | 987.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 987.45 | 986.72 | 987.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 987.45 | 986.72 | 987.14 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 12:15:00 | 989.90 | 987.63 | 987.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 1011.05 | 993.03 | 990.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 15:15:00 | 994.70 | 996.29 | 993.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 15:15:00 | 994.70 | 996.29 | 993.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 994.70 | 996.29 | 993.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:15:00 | 985.45 | 996.29 | 993.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 988.20 | 994.67 | 992.95 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 985.65 | 990.71 | 991.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 983.20 | 989.21 | 990.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 994.25 | 988.44 | 989.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 10:15:00 | 994.25 | 988.44 | 989.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 994.25 | 988.44 | 989.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 10:30:00 | 995.20 | 988.44 | 989.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 990.45 | 988.84 | 989.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 987.80 | 988.84 | 989.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 13:15:00 | 994.90 | 990.20 | 990.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 994.90 | 990.20 | 990.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 1001.35 | 992.43 | 991.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 1000.00 | 1000.10 | 995.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 11:30:00 | 999.95 | 1000.10 | 995.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 999.00 | 999.88 | 996.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 999.30 | 999.88 | 996.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 989.70 | 997.84 | 995.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 989.70 | 997.84 | 995.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 991.15 | 996.50 | 995.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:15:00 | 991.95 | 996.50 | 995.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 991.95 | 995.59 | 994.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 997.50 | 995.59 | 994.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:30:00 | 994.35 | 996.87 | 995.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 1002.60 | 995.57 | 995.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 994.35 | 995.32 | 995.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 994.35 | 995.32 | 995.36 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 997.00 | 995.66 | 995.51 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 12:15:00 | 992.05 | 994.84 | 995.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 14:15:00 | 986.00 | 992.61 | 994.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 12:15:00 | 996.50 | 987.97 | 990.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 12:15:00 | 996.50 | 987.97 | 990.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 996.50 | 987.97 | 990.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 996.50 | 987.97 | 990.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 989.70 | 988.31 | 990.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 984.90 | 989.82 | 990.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 12:15:00 | 987.15 | 990.53 | 991.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 985.00 | 988.69 | 989.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:00:00 | 986.75 | 984.74 | 986.78 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 986.10 | 985.01 | 986.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:30:00 | 987.70 | 985.01 | 986.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 985.85 | 985.18 | 986.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 985.85 | 985.18 | 986.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 983.05 | 984.75 | 986.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 15:00:00 | 982.05 | 985.27 | 986.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 10:15:00 | 989.00 | 985.74 | 986.22 | SL hit (close>static) qty=1.00 sl=988.95 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 11:15:00 | 1003.75 | 985.64 | 984.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 1039.10 | 1012.01 | 1001.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 15:15:00 | 1024.00 | 1025.32 | 1014.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 1016.95 | 1025.32 | 1014.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1004.90 | 1021.23 | 1013.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 1004.90 | 1021.23 | 1013.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1007.60 | 1018.51 | 1013.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:15:00 | 1005.00 | 1018.51 | 1013.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 1002.80 | 1015.37 | 1012.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:00:00 | 1002.80 | 1015.37 | 1012.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 999.05 | 1009.67 | 1010.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 990.10 | 1005.75 | 1008.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 13:15:00 | 878.50 | 873.50 | 883.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:00:00 | 878.50 | 873.50 | 883.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 889.00 | 876.53 | 882.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 889.00 | 876.53 | 882.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 906.90 | 882.61 | 884.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 914.95 | 882.61 | 884.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 899.75 | 886.04 | 885.69 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 886.00 | 891.25 | 891.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 09:15:00 | 877.40 | 885.44 | 888.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 11:15:00 | 891.80 | 886.69 | 888.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 11:15:00 | 891.80 | 886.69 | 888.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 891.80 | 886.69 | 888.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:00:00 | 891.80 | 886.69 | 888.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 891.85 | 887.72 | 888.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:30:00 | 891.50 | 887.72 | 888.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 890.90 | 888.36 | 889.01 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 15:15:00 | 893.40 | 889.95 | 889.66 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 875.55 | 887.75 | 888.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 14:15:00 | 870.25 | 877.78 | 881.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 744.00 | 739.06 | 752.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:15:00 | 741.45 | 739.06 | 752.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 748.05 | 740.86 | 752.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 11:00:00 | 735.45 | 749.46 | 753.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 14:15:00 | 734.05 | 744.59 | 749.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 15:15:00 | 733.05 | 743.47 | 748.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 735.10 | 742.70 | 746.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 742.70 | 738.59 | 741.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 742.70 | 738.59 | 741.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 742.00 | 739.27 | 741.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 735.35 | 738.82 | 741.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 740.00 | 741.17 | 741.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 772.20 | 747.19 | 744.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 772.20 | 747.19 | 744.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 787.10 | 755.17 | 748.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 777.90 | 778.73 | 768.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 767.75 | 775.14 | 769.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 767.75 | 775.14 | 769.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 763.80 | 775.14 | 769.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 765.30 | 773.17 | 769.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 765.70 | 773.17 | 769.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 749.50 | 764.72 | 766.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 744.85 | 755.46 | 760.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 726.10 | 724.79 | 733.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:45:00 | 726.15 | 724.79 | 733.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 726.95 | 724.09 | 730.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 729.50 | 724.09 | 730.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 719.30 | 722.78 | 728.02 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 731.20 | 727.15 | 726.64 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 717.70 | 725.56 | 726.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 704.95 | 719.85 | 723.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 713.45 | 710.89 | 715.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 713.45 | 710.89 | 715.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 710.35 | 710.78 | 715.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:15:00 | 714.00 | 710.78 | 715.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 711.55 | 710.93 | 715.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 12:15:00 | 709.05 | 710.93 | 715.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 13:30:00 | 709.80 | 710.48 | 714.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 14:30:00 | 708.30 | 710.19 | 713.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 726.25 | 713.21 | 714.43 | SL hit (close>static) qty=1.00 sl=720.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 730.80 | 716.73 | 715.91 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 718.10 | 723.09 | 723.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 09:15:00 | 700.00 | 717.46 | 720.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 716.25 | 710.21 | 715.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 716.25 | 710.21 | 715.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 716.25 | 710.21 | 715.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 718.85 | 710.21 | 715.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 715.45 | 711.25 | 715.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 727.45 | 711.25 | 715.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 729.25 | 714.85 | 716.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 732.50 | 714.85 | 716.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 735.00 | 718.88 | 718.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 12:15:00 | 740.75 | 725.99 | 721.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 782.10 | 783.29 | 773.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:45:00 | 779.15 | 783.29 | 773.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 829.60 | 846.82 | 839.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 829.60 | 846.82 | 839.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 830.00 | 843.45 | 838.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 827.70 | 843.45 | 838.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 836.10 | 841.04 | 837.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:45:00 | 838.75 | 841.04 | 837.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 830.35 | 838.90 | 837.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 830.35 | 838.90 | 837.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 830.05 | 837.13 | 836.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 834.00 | 837.13 | 836.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 09:30:00 | 836.60 | 837.39 | 836.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 835.15 | 836.32 | 836.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 11:15:00 | 835.15 | 836.32 | 836.36 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 838.55 | 836.71 | 836.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 853.00 | 840.06 | 838.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 14:15:00 | 837.70 | 842.64 | 840.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 14:15:00 | 837.70 | 842.64 | 840.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 837.70 | 842.64 | 840.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 836.85 | 842.64 | 840.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 834.05 | 840.93 | 839.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 834.65 | 840.93 | 839.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 842.55 | 840.53 | 839.87 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 14:15:00 | 836.25 | 839.72 | 839.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 831.15 | 837.89 | 838.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 835.85 | 834.68 | 836.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 14:15:00 | 835.85 | 834.68 | 836.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 835.85 | 834.68 | 836.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 835.85 | 834.68 | 836.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 832.00 | 834.14 | 836.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 840.80 | 834.14 | 836.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 831.00 | 833.51 | 835.70 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 09:15:00 | 894.45 | 840.44 | 836.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 10:15:00 | 906.30 | 853.61 | 843.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 12:15:00 | 907.60 | 917.40 | 892.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-24 13:00:00 | 907.60 | 917.40 | 892.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 980.00 | 1001.31 | 982.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 980.00 | 1001.31 | 982.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 968.25 | 994.70 | 981.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 14:45:00 | 983.30 | 977.83 | 976.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 15:15:00 | 961.90 | 974.64 | 975.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 961.90 | 974.64 | 975.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 938.60 | 967.43 | 971.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 14:15:00 | 958.50 | 955.02 | 962.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 14:15:00 | 958.50 | 955.02 | 962.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 958.50 | 955.02 | 962.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 957.05 | 955.02 | 962.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 965.20 | 957.38 | 962.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 10:15:00 | 958.00 | 957.38 | 962.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 15:15:00 | 971.50 | 964.77 | 964.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 971.50 | 964.77 | 964.37 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 950.15 | 961.84 | 963.07 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 987.80 | 963.22 | 962.10 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 942.00 | 964.00 | 964.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 930.55 | 957.31 | 961.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 997.50 | 944.78 | 950.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 997.50 | 944.78 | 950.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 997.50 | 944.78 | 950.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 997.50 | 944.78 | 950.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 968.95 | 949.61 | 952.43 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 974.85 | 954.66 | 954.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 12:15:00 | 990.50 | 961.83 | 957.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 949.15 | 966.72 | 962.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 949.15 | 966.72 | 962.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 949.15 | 966.72 | 962.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 952.30 | 966.72 | 962.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 937.00 | 960.77 | 960.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 927.40 | 960.77 | 960.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 930.35 | 954.69 | 957.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 13:15:00 | 927.10 | 945.54 | 952.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 10:15:00 | 914.00 | 906.10 | 914.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 10:15:00 | 914.00 | 906.10 | 914.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 914.00 | 906.10 | 914.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:45:00 | 914.00 | 906.10 | 914.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 902.80 | 905.44 | 913.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 14:15:00 | 894.25 | 903.68 | 911.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 898.10 | 901.19 | 907.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:00:00 | 899.30 | 900.91 | 906.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:45:00 | 898.70 | 900.62 | 906.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 898.95 | 899.85 | 903.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 912.70 | 906.11 | 905.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 912.70 | 906.11 | 905.48 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 09:15:00 | 890.90 | 903.52 | 904.91 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 920.40 | 901.49 | 901.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 921.45 | 909.09 | 905.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 910.80 | 912.06 | 907.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 910.80 | 912.06 | 907.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 908.10 | 911.26 | 907.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 905.15 | 911.26 | 907.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 907.15 | 910.44 | 907.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 910.25 | 910.44 | 907.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 907.60 | 909.87 | 907.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 908.55 | 909.87 | 907.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 904.20 | 908.74 | 907.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:45:00 | 903.95 | 908.74 | 907.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 901.60 | 907.31 | 906.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 902.15 | 907.31 | 906.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 903.75 | 906.60 | 906.60 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 882.85 | 901.85 | 904.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 876.15 | 896.71 | 901.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 888.35 | 886.22 | 893.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 893.20 | 886.22 | 893.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 926.70 | 894.32 | 896.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 926.70 | 894.32 | 896.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 918.05 | 899.06 | 898.52 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 893.55 | 901.16 | 901.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 888.70 | 898.67 | 900.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 14:15:00 | 858.00 | 856.44 | 867.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 15:00:00 | 858.00 | 856.44 | 867.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 857.10 | 856.31 | 865.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 865.30 | 856.31 | 865.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 811.80 | 808.90 | 817.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 803.45 | 809.05 | 816.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 823.65 | 810.70 | 815.57 | SL hit (close>static) qty=1.00 sl=818.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 849.85 | 815.67 | 811.53 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 810.00 | 817.15 | 817.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 797.90 | 810.05 | 813.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 746.95 | 743.75 | 758.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 09:15:00 | 729.30 | 740.01 | 750.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 692.83 | 710.71 | 727.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 704.95 | 701.54 | 715.28 | SL hit (close>ema200) qty=0.50 sl=701.54 alert=retest1 |

### Cycle 65 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 716.70 | 706.47 | 705.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 722.80 | 711.97 | 708.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 713.30 | 714.43 | 711.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 713.30 | 714.43 | 711.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 713.30 | 714.43 | 711.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:00:00 | 719.85 | 715.52 | 711.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:45:00 | 719.30 | 721.54 | 716.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 13:15:00 | 712.80 | 720.86 | 721.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 712.80 | 720.86 | 721.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 701.90 | 715.26 | 718.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 10:15:00 | 707.15 | 704.48 | 709.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 10:15:00 | 707.15 | 704.48 | 709.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 707.15 | 704.48 | 709.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 11:45:00 | 700.40 | 703.71 | 708.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:30:00 | 700.15 | 701.02 | 707.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 14:15:00 | 665.38 | 691.50 | 701.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 14:15:00 | 665.14 | 691.50 | 701.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 647.35 | 646.72 | 660.15 | SL hit (close>ema200) qty=0.50 sl=646.72 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 676.60 | 664.29 | 662.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 11:15:00 | 678.60 | 667.15 | 664.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 674.45 | 677.52 | 672.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 674.45 | 677.52 | 672.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 669.35 | 675.88 | 672.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 669.35 | 675.88 | 672.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 671.50 | 675.01 | 672.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:45:00 | 676.70 | 673.95 | 671.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 677.90 | 673.36 | 671.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 674.75 | 673.00 | 671.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 12:15:00 | 667.65 | 671.96 | 671.61 | SL hit (close<static) qty=1.00 sl=668.70 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 662.05 | 669.97 | 670.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 653.40 | 666.66 | 669.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 671.20 | 664.90 | 667.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 671.20 | 664.90 | 667.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 671.20 | 664.90 | 667.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 671.20 | 664.90 | 667.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 679.15 | 667.75 | 668.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 679.15 | 667.75 | 668.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 664.90 | 667.18 | 668.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:45:00 | 684.25 | 667.18 | 668.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 668.30 | 664.64 | 666.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 668.70 | 664.64 | 666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 666.00 | 664.92 | 666.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 654.00 | 664.92 | 666.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 653.65 | 662.66 | 665.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 651.75 | 661.08 | 664.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 651.70 | 659.50 | 663.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 650.45 | 656.38 | 659.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 651.15 | 655.61 | 658.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 650.75 | 644.26 | 648.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 650.75 | 644.26 | 648.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 649.00 | 645.21 | 648.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 650.00 | 645.21 | 648.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 649.65 | 646.10 | 649.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:30:00 | 649.75 | 646.10 | 649.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 650.40 | 646.96 | 649.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 650.40 | 646.96 | 649.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 653.80 | 648.33 | 649.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:45:00 | 654.20 | 648.33 | 649.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 656.90 | 650.04 | 650.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 656.90 | 650.04 | 650.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 654.60 | 650.95 | 650.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 654.60 | 650.95 | 650.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 683.50 | 657.46 | 653.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 686.00 | 686.31 | 677.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 692.30 | 686.31 | 677.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 699.80 | 689.01 | 679.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 701.50 | 691.69 | 681.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:30:00 | 703.00 | 694.35 | 683.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 11:15:00 | 710.85 | 718.16 | 718.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 710.85 | 718.16 | 718.86 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 728.20 | 720.65 | 719.66 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 10:15:00 | 713.20 | 718.16 | 718.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 11:15:00 | 713.05 | 717.14 | 718.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 14:15:00 | 685.75 | 684.15 | 691.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 14:45:00 | 684.75 | 684.15 | 691.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 687.75 | 684.37 | 690.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:00:00 | 673.00 | 685.67 | 689.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 605.70 | 664.90 | 675.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 658.10 | 636.58 | 636.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 704.55 | 659.34 | 647.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 12:15:00 | 749.60 | 750.34 | 730.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 13:00:00 | 749.60 | 750.34 | 730.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 763.65 | 774.21 | 764.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 763.65 | 774.21 | 764.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 770.00 | 773.37 | 765.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 778.10 | 773.37 | 765.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:00:00 | 775.50 | 791.97 | 790.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 13:15:00 | 781.00 | 788.18 | 789.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 781.00 | 788.18 | 789.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 764.95 | 783.53 | 786.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 784.40 | 770.86 | 775.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 784.40 | 770.86 | 775.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 784.40 | 770.86 | 775.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 786.20 | 770.86 | 775.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 779.85 | 772.66 | 776.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 779.35 | 772.66 | 776.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 793.45 | 777.83 | 778.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:45:00 | 794.65 | 777.83 | 778.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 795.00 | 781.26 | 779.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 813.00 | 790.16 | 784.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 793.70 | 795.62 | 789.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 13:30:00 | 794.80 | 795.62 | 789.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 786.05 | 792.84 | 789.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:45:00 | 790.95 | 794.71 | 790.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:30:00 | 791.80 | 793.59 | 791.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 801.10 | 792.45 | 791.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 789.50 | 799.98 | 800.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 789.50 | 799.98 | 800.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 783.00 | 796.59 | 798.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 799.60 | 794.65 | 797.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 799.60 | 794.65 | 797.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 799.60 | 794.65 | 797.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 799.60 | 794.65 | 797.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 824.85 | 800.69 | 799.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 840.00 | 815.91 | 808.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 844.75 | 845.08 | 828.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 844.75 | 845.08 | 828.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 846.15 | 844.52 | 830.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:30:00 | 869.85 | 845.08 | 834.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 13:45:00 | 855.15 | 847.39 | 836.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 866.80 | 844.56 | 837.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-12 12:15:00 | 940.67 | 880.13 | 857.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 1172.00 | 1181.95 | 1182.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 1169.60 | 1177.90 | 1180.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1172.60 | 1156.13 | 1163.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1170.10 | 1158.93 | 1164.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:30:00 | 1172.60 | 1158.93 | 1164.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1173.50 | 1164.49 | 1165.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 1173.50 | 1164.49 | 1165.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 1174.70 | 1166.53 | 1166.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 1226.50 | 1179.56 | 1172.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 1203.60 | 1204.32 | 1192.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 12:30:00 | 1211.90 | 1204.71 | 1194.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1187.00 | 1201.17 | 1193.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1187.00 | 1201.17 | 1193.78 | SL hit (close<ema400) qty=1.00 sl=1193.78 alert=retest1 |

### Cycle 80 — SELL (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 14:15:00 | 1187.60 | 1193.30 | 1193.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1164.20 | 1186.81 | 1190.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1183.50 | 1185.73 | 1188.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1199.90 | 1188.56 | 1189.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 1199.80 | 1188.56 | 1189.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1213.90 | 1193.63 | 1191.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1238.20 | 1204.86 | 1197.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 1216.10 | 1221.52 | 1212.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1212.70 | 1219.76 | 1212.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 1212.70 | 1219.76 | 1212.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1204.90 | 1216.79 | 1212.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 1204.00 | 1216.79 | 1212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1199.80 | 1213.39 | 1211.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:15:00 | 1194.10 | 1213.39 | 1211.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1198.00 | 1207.85 | 1208.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1170.80 | 1200.44 | 1205.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 13:15:00 | 1071.70 | 1069.72 | 1092.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 1071.60 | 1069.72 | 1092.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1088.30 | 1075.78 | 1089.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:15:00 | 1090.50 | 1075.78 | 1089.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1087.30 | 1078.09 | 1089.61 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1116.80 | 1096.28 | 1095.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 1135.70 | 1107.01 | 1100.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 1119.10 | 1119.31 | 1110.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1119.10 | 1119.31 | 1110.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1111.50 | 1118.75 | 1112.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1111.50 | 1118.75 | 1112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1118.80 | 1118.76 | 1113.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 1134.30 | 1127.58 | 1118.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:45:00 | 1140.30 | 1131.33 | 1123.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1139.00 | 1140.21 | 1137.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 1131.90 | 1139.71 | 1140.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1131.90 | 1139.71 | 1140.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1125.60 | 1136.89 | 1138.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1144.30 | 1133.18 | 1135.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1139.10 | 1134.36 | 1135.97 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 1149.40 | 1139.34 | 1138.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1161.80 | 1148.70 | 1143.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1163.00 | 1173.13 | 1164.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1154.10 | 1169.32 | 1163.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 1151.00 | 1169.32 | 1163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 1147.40 | 1159.82 | 1159.90 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 1173.30 | 1158.87 | 1157.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 1177.50 | 1166.31 | 1161.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 1168.60 | 1171.53 | 1165.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1177.40 | 1181.36 | 1176.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 1177.30 | 1181.36 | 1176.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1175.10 | 1180.11 | 1176.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1174.00 | 1180.11 | 1176.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1188.10 | 1181.71 | 1177.14 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1171.70 | 1178.42 | 1178.90 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1199.10 | 1179.18 | 1177.38 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1147.80 | 1179.38 | 1181.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 1139.60 | 1171.43 | 1177.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 1081.50 | 1079.75 | 1114.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:30:00 | 1089.60 | 1079.75 | 1114.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1045.50 | 1041.56 | 1061.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1041.00 | 1041.56 | 1061.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 1063.30 | 1050.86 | 1059.97 | SL hit (close>static) qty=1.00 sl=1061.90 alert=retest2 |

### Cycle 91 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 941.20 | 932.36 | 931.74 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 909.10 | 928.61 | 930.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 12:15:00 | 905.70 | 918.62 | 924.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 925.00 | 914.94 | 920.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 937.20 | 919.40 | 922.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 937.20 | 919.40 | 922.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 953.90 | 926.30 | 924.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 963.00 | 933.64 | 928.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 966.10 | 968.36 | 954.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 966.10 | 968.36 | 954.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 969.20 | 966.87 | 956.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 955.40 | 966.87 | 956.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 982.40 | 972.31 | 965.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 997.40 | 978.80 | 969.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 988.50 | 980.84 | 977.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 15:15:00 | 965.50 | 975.56 | 976.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 965.50 | 975.56 | 976.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 964.00 | 973.25 | 975.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 979.30 | 974.46 | 975.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 975.00 | 974.57 | 975.76 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 980.90 | 976.82 | 976.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 990.05 | 981.13 | 978.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 981.10 | 987.05 | 983.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 988.45 | 987.33 | 983.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 982.05 | 987.33 | 983.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 982.05 | 986.27 | 983.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 975.85 | 986.27 | 983.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 971.80 | 983.38 | 982.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 969.40 | 983.38 | 982.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 967.35 | 980.17 | 981.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 11:15:00 | 965.65 | 977.27 | 979.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 933.70 | 913.96 | 921.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 941.00 | 919.37 | 923.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 943.95 | 919.37 | 923.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 949.50 | 929.49 | 927.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 968.00 | 937.20 | 931.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1020.75 | 1020.98 | 1005.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1020.75 | 1020.98 | 1005.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1019.80 | 1021.56 | 1015.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 1016.10 | 1021.56 | 1015.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1030.15 | 1032.98 | 1026.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 1030.20 | 1032.98 | 1026.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1025.65 | 1031.52 | 1026.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 1025.65 | 1031.52 | 1026.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1027.05 | 1030.62 | 1026.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1023.45 | 1030.62 | 1026.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1027.25 | 1029.95 | 1026.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1052.00 | 1029.95 | 1026.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1052.05 | 1058.81 | 1059.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1052.05 | 1058.81 | 1059.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1038.00 | 1052.90 | 1056.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1075.15 | 1046.46 | 1050.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1063.85 | 1049.94 | 1051.80 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 1068.35 | 1053.62 | 1053.31 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 1009.75 | 1049.56 | 1052.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 1005.00 | 1040.65 | 1048.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 976.00 | 974.16 | 995.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 976.00 | 974.16 | 995.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 975.15 | 974.38 | 980.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 976.90 | 974.38 | 980.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 976.65 | 974.76 | 979.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 984.65 | 974.76 | 979.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 984.00 | 976.78 | 978.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 984.00 | 976.78 | 978.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 983.75 | 978.17 | 979.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 979.00 | 978.17 | 979.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 988.15 | 981.36 | 980.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 988.15 | 981.36 | 980.56 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 973.75 | 980.27 | 980.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 971.50 | 978.52 | 979.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 982.95 | 979.98 | 980.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 985.75 | 981.14 | 980.86 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 975.70 | 980.94 | 981.02 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 988.80 | 981.20 | 981.01 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 969.25 | 978.81 | 979.94 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 993.20 | 980.91 | 979.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1009.10 | 986.55 | 982.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 990.30 | 998.41 | 990.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 990.00 | 996.73 | 990.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 990.00 | 996.73 | 990.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 988.40 | 995.06 | 990.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 988.55 | 995.06 | 990.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 988.05 | 993.66 | 990.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 988.05 | 993.66 | 990.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 977.00 | 986.90 | 987.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 962.85 | 982.09 | 985.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 997.05 | 971.10 | 976.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 992.25 | 975.33 | 977.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 995.00 | 975.33 | 977.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1000.15 | 980.29 | 979.95 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 971.10 | 981.39 | 981.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 966.60 | 974.24 | 977.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 973.90 | 957.19 | 961.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 969.60 | 959.67 | 962.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 959.40 | 959.67 | 962.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 964.10 | 960.60 | 962.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 964.10 | 960.60 | 962.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 971.15 | 962.71 | 963.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 971.15 | 962.71 | 963.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 975.15 | 965.20 | 964.26 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 956.40 | 962.55 | 963.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 954.50 | 960.94 | 962.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 963.35 | 961.00 | 962.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 13:15:00 | 992.55 | 967.31 | 964.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 998.10 | 973.47 | 968.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1026.00 | 1026.21 | 1007.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:30:00 | 1025.90 | 1026.21 | 1007.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1017.80 | 1021.95 | 1013.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1043.80 | 1025.26 | 1015.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1054.35 | 1027.38 | 1020.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1050.30 | 1033.56 | 1024.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1045.55 | 1037.57 | 1028.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 09:15:00 | 1148.18 | 1122.88 | 1085.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1131.30 | 1148.74 | 1149.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1123.30 | 1143.65 | 1146.81 | Break + close below crossover candle low |

### Cycle 115 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 1156.00 | 1110.80 | 1110.70 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1087.40 | 1115.12 | 1117.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1079.10 | 1093.50 | 1099.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 1105.90 | 1095.82 | 1099.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1100.80 | 1096.81 | 1099.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 1108.80 | 1096.81 | 1099.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1099.70 | 1097.39 | 1099.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1099.70 | 1097.39 | 1099.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1087.50 | 1095.41 | 1098.47 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1137.90 | 1102.72 | 1101.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 10:15:00 | 1158.50 | 1113.88 | 1106.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1099.60 | 1131.35 | 1121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1091.30 | 1123.34 | 1118.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1091.30 | 1123.34 | 1118.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 1095.90 | 1113.62 | 1114.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1078.20 | 1099.70 | 1107.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 11:15:00 | 1093.80 | 1092.18 | 1098.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1093.80 | 1092.18 | 1098.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1098.30 | 1093.40 | 1098.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1098.40 | 1093.40 | 1098.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1099.50 | 1094.62 | 1098.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1099.50 | 1094.62 | 1098.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1095.50 | 1094.80 | 1097.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 1090.30 | 1094.80 | 1097.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1092.80 | 1092.43 | 1095.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 1092.70 | 1093.19 | 1095.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1101.00 | 1093.36 | 1094.83 | SL hit (close>static) qty=1.00 sl=1100.10 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1103.30 | 1095.77 | 1095.69 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1092.70 | 1095.83 | 1096.04 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1110.00 | 1098.66 | 1097.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 1132.30 | 1112.91 | 1106.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 1113.90 | 1114.24 | 1108.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:45:00 | 1114.60 | 1114.24 | 1108.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1109.70 | 1113.33 | 1108.63 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1095.00 | 1105.72 | 1106.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 1091.50 | 1102.88 | 1105.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1094.20 | 1098.53 | 1101.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 1090.20 | 1096.32 | 1100.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1109.30 | 1100.91 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1109.30 | 1100.91 | 1100.85 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 1091.80 | 1099.60 | 1100.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1080.00 | 1094.11 | 1097.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1031.00 | 1030.18 | 1046.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:15:00 | 1035.40 | 1030.18 | 1046.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1044.20 | 1037.34 | 1045.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1028.90 | 1036.17 | 1044.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 1033.00 | 1028.48 | 1033.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 1029.00 | 1029.92 | 1033.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1029.50 | 1031.94 | 1034.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1032.90 | 1032.30 | 1034.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 1031.70 | 1032.30 | 1034.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1030.30 | 1031.90 | 1033.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1027.00 | 1030.82 | 1033.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1025.50 | 1030.18 | 1032.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 1027.10 | 1029.32 | 1031.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1025.00 | 1029.13 | 1030.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1020.00 | 1027.30 | 1029.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:00:00 | 1012.10 | 1021.85 | 1026.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 1015.90 | 1020.68 | 1025.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1013.30 | 1021.92 | 1023.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1054.50 | 1029.74 | 1026.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1054.50 | 1029.74 | 1026.67 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 1027.00 | 1032.68 | 1033.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 1024.20 | 1030.48 | 1032.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 1007.00 | 1000.45 | 1008.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 995.30 | 999.42 | 1007.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 994.70 | 999.42 | 1007.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 994.70 | 998.54 | 1006.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 993.10 | 998.54 | 1006.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 990.40 | 997.30 | 1003.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 944.97 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 944.97 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 943.44 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 940.88 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 970.10 | 962.97 | 974.16 | SL hit (close>ema200) qty=0.50 sl=962.97 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 967.60 | 962.49 | 962.31 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 958.00 | 962.60 | 962.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 948.80 | 958.24 | 960.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 972.00 | 957.47 | 959.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 976.85 | 961.35 | 960.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 987.05 | 970.39 | 965.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 962.85 | 975.72 | 970.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 969.65 | 974.51 | 970.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 966.10 | 974.51 | 970.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 951.80 | 967.44 | 967.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 945.35 | 963.02 | 965.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 919.75 | 916.03 | 928.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 925.35 | 916.03 | 928.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 936.60 | 920.14 | 929.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 936.60 | 920.14 | 929.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 940.90 | 924.29 | 930.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 940.90 | 924.29 | 930.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 930.65 | 929.21 | 931.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 930.65 | 929.21 | 931.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 929.00 | 929.17 | 931.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 930.10 | 929.17 | 931.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 917.05 | 926.75 | 929.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 904.35 | 923.58 | 925.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 911.75 | 901.53 | 903.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 913.40 | 905.32 | 905.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 913.40 | 905.32 | 905.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 917.10 | 907.68 | 906.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 902.60 | 913.15 | 909.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 908.55 | 912.23 | 909.69 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 892.65 | 905.50 | 906.89 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 907.50 | 901.56 | 900.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 908.40 | 903.70 | 902.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 888.00 | 901.63 | 901.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 889.00 | 899.10 | 900.35 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 919.15 | 903.15 | 901.40 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 834.00 | 896.61 | 900.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 777.25 | 796.01 | 811.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 766.20 | 764.41 | 775.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 767.10 | 764.41 | 775.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 775.95 | 766.72 | 775.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 775.95 | 766.72 | 775.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 773.85 | 768.15 | 775.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:30:00 | 774.60 | 768.15 | 775.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 773.60 | 769.24 | 775.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:00:00 | 773.60 | 769.24 | 775.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 775.90 | 771.05 | 774.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 767.80 | 770.40 | 774.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 729.41 | 740.81 | 753.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 691.02 | 712.24 | 730.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 137 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 736.30 | 704.71 | 701.68 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 692.00 | 701.96 | 702.87 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 704.50 | 703.35 | 703.29 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 702.50 | 703.18 | 703.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 700.20 | 702.58 | 702.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 702.70 | 702.87 | 703.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 700.50 | 702.39 | 702.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 697.00 | 702.39 | 702.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 704.60 | 701.91 | 702.34 | SL hit (close>static) qty=1.00 sl=704.20 alert=retest2 |

### Cycle 141 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 700.15 | 697.29 | 697.10 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 692.35 | 697.13 | 697.15 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 10:15:00 | 697.55 | 697.22 | 697.19 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 690.80 | 695.93 | 696.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 684.00 | 693.40 | 695.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 686.00 | 685.05 | 689.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 686.00 | 685.05 | 689.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 679.00 | 664.99 | 670.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 679.00 | 664.99 | 670.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 684.45 | 668.88 | 671.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 684.45 | 668.88 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 688.60 | 675.73 | 674.60 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 665.30 | 673.46 | 674.48 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 682.00 | 674.76 | 674.60 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 669.35 | 674.06 | 674.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 666.80 | 672.61 | 673.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 668.25 | 664.90 | 668.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 668.25 | 664.90 | 668.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 669.35 | 665.79 | 668.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 669.35 | 665.79 | 668.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 671.00 | 666.83 | 668.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 671.95 | 666.83 | 668.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 667.15 | 666.89 | 668.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 667.70 | 666.89 | 668.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 664.35 | 665.05 | 667.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 667.20 | 665.05 | 667.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 660.65 | 663.46 | 665.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 664.05 | 663.46 | 665.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 655.55 | 661.48 | 664.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 647.45 | 661.48 | 664.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 11:00:00 | 647.45 | 658.68 | 662.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 11:45:00 | 647.95 | 656.75 | 661.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:30:00 | 646.35 | 654.69 | 660.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 646.70 | 649.48 | 655.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 642.65 | 647.66 | 652.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 676.90 | 652.66 | 653.88 | SL hit (close>static) qty=1.00 sl=666.75 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 692.40 | 660.61 | 657.38 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 658.75 | 662.89 | 663.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 657.50 | 661.34 | 662.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 647.50 | 641.54 | 647.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 648.00 | 642.83 | 647.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 647.05 | 642.83 | 647.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 646.00 | 643.46 | 647.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 648.05 | 643.46 | 647.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 643.35 | 643.44 | 647.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 644.25 | 643.44 | 647.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 657.50 | 646.82 | 647.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 657.50 | 646.82 | 647.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 662.25 | 649.91 | 649.27 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 637.80 | 649.47 | 650.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 634.45 | 644.19 | 647.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 639.85 | 616.33 | 626.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 635.20 | 620.10 | 626.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 639.20 | 620.10 | 626.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 634.70 | 630.66 | 630.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 638.70 | 632.27 | 631.36 | Break + close above crossover candle high |

### Cycle 154 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 621.10 | 630.03 | 630.42 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 646.20 | 633.13 | 631.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 658.55 | 638.21 | 634.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 675.50 | 679.03 | 670.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 675.50 | 679.03 | 670.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 673.75 | 678.08 | 671.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 673.75 | 678.08 | 671.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 669.85 | 676.43 | 671.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 669.85 | 676.43 | 671.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 671.95 | 675.53 | 671.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 669.00 | 675.53 | 671.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 673.40 | 675.11 | 671.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 672.90 | 675.11 | 671.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 680.60 | 676.21 | 672.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 685.65 | 678.24 | 673.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 684.70 | 682.33 | 677.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 646.20 | 675.48 | 675.36 | SL hit (close<static) qty=1.00 sl=670.00 alert=retest2 |

### Cycle 156 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 657.30 | 671.85 | 673.71 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 681.85 | 669.55 | 668.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 690.00 | 673.64 | 670.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 13:15:00 | 694.10 | 695.79 | 689.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 13:30:00 | 695.35 | 695.79 | 689.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 684.30 | 693.17 | 690.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 679.05 | 693.17 | 690.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 685.00 | 691.54 | 689.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 685.00 | 691.54 | 689.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 681.05 | 688.79 | 688.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 681.05 | 688.79 | 688.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 678.75 | 686.78 | 687.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 665.70 | 677.52 | 681.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 695.00 | 677.87 | 680.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 694.65 | 681.23 | 681.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:15:00 | 697.50 | 681.23 | 681.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 697.50 | 684.48 | 682.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 703.70 | 688.32 | 684.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 683.65 | 691.47 | 691.52 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 712.25 | 694.50 | 692.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 725.05 | 708.94 | 702.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 738.95 | 742.51 | 730.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 738.95 | 742.51 | 730.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 740.70 | 741.73 | 733.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 755.00 | 742.65 | 740.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 14:15:00 | 830.50 | 795.53 | 784.79 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 10:30:00 | 877.95 | 2024-05-18 09:15:00 | 898.95 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-05-22 09:45:00 | 899.55 | 2024-05-28 09:15:00 | 897.25 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-05-22 11:45:00 | 899.00 | 2024-05-28 09:15:00 | 897.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-05-22 15:15:00 | 899.35 | 2024-05-28 09:15:00 | 897.25 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-05-23 14:30:00 | 899.05 | 2024-05-28 09:15:00 | 897.25 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-06-19 09:15:00 | 1058.75 | 2024-06-19 12:15:00 | 1048.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-19 09:45:00 | 1060.30 | 2024-06-19 12:15:00 | 1048.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-06-19 10:30:00 | 1060.55 | 2024-06-19 12:15:00 | 1048.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-07-03 15:00:00 | 1094.75 | 2024-07-08 09:15:00 | 1083.35 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-04 10:30:00 | 1098.35 | 2024-07-08 09:15:00 | 1083.35 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-07-04 13:30:00 | 1096.70 | 2024-07-08 09:15:00 | 1083.35 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-04 14:30:00 | 1096.80 | 2024-07-08 09:15:00 | 1083.35 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-07-26 09:15:00 | 991.00 | 2024-08-02 09:15:00 | 943.63 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2024-07-26 11:00:00 | 993.30 | 2024-08-02 09:15:00 | 943.87 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-07-29 09:15:00 | 981.20 | 2024-08-05 09:15:00 | 941.45 | PARTIAL | 0.50 | 4.05% |
| SELL | retest2 | 2024-07-30 09:15:00 | 993.55 | 2024-08-05 09:15:00 | 932.14 | PARTIAL | 0.50 | 6.18% |
| SELL | retest2 | 2024-07-31 09:15:00 | 983.50 | 2024-08-05 09:15:00 | 934.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 10:00:00 | 982.45 | 2024-08-05 09:15:00 | 933.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 12:45:00 | 983.40 | 2024-08-05 09:15:00 | 934.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-26 09:15:00 | 991.00 | 2024-08-05 15:15:00 | 891.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-26 11:00:00 | 993.30 | 2024-08-05 15:15:00 | 893.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-29 09:15:00 | 981.20 | 2024-08-05 15:15:00 | 894.19 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2024-07-30 09:15:00 | 993.55 | 2024-08-06 09:15:00 | 926.00 | STOP_HIT | 0.50 | 6.80% |
| SELL | retest2 | 2024-07-31 09:15:00 | 983.50 | 2024-08-06 09:15:00 | 926.00 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2024-07-31 10:00:00 | 982.45 | 2024-08-06 09:15:00 | 926.00 | STOP_HIT | 0.50 | 5.75% |
| SELL | retest2 | 2024-07-31 12:45:00 | 983.40 | 2024-08-06 09:15:00 | 926.00 | STOP_HIT | 0.50 | 5.84% |
| BUY | retest2 | 2024-08-09 09:15:00 | 940.70 | 2024-08-12 09:15:00 | 928.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-09 14:00:00 | 939.15 | 2024-08-12 09:15:00 | 928.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-09 14:30:00 | 940.00 | 2024-08-12 09:15:00 | 928.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-08-09 15:15:00 | 939.00 | 2024-08-12 09:15:00 | 928.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2024-08-21 09:15:00 | 1000.85 | 2024-08-21 15:15:00 | 982.70 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest1 | 2024-08-21 09:45:00 | 998.90 | 2024-08-21 15:15:00 | 982.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-09-10 12:15:00 | 987.80 | 2024-09-10 13:15:00 | 994.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-12 09:15:00 | 997.50 | 2024-09-13 09:15:00 | 994.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-09-12 11:30:00 | 994.35 | 2024-09-13 09:15:00 | 994.35 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-09-13 09:15:00 | 1002.60 | 2024-09-13 09:15:00 | 994.35 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-09-17 09:15:00 | 984.90 | 2024-09-20 10:15:00 | 989.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-09-17 12:15:00 | 987.15 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-09-18 10:00:00 | 985.00 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-09-19 10:00:00 | 986.75 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-09-19 15:00:00 | 982.05 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-09-20 13:45:00 | 982.95 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-09-20 14:15:00 | 981.25 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-09-24 10:30:00 | 982.15 | 2024-09-24 11:15:00 | 1003.75 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-10-31 11:00:00 | 735.45 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2024-10-31 14:15:00 | 734.05 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2024-10-31 15:15:00 | 733.05 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2024-11-04 09:15:00 | 735.10 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2024-11-05 11:30:00 | 735.35 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2024-11-05 15:15:00 | 740.00 | 2024-11-06 09:15:00 | 772.20 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2024-11-22 12:15:00 | 709.05 | 2024-11-25 09:15:00 | 726.25 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-11-22 13:30:00 | 709.80 | 2024-11-25 09:15:00 | 726.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-11-22 14:30:00 | 708.30 | 2024-11-25 09:15:00 | 726.25 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2024-12-13 15:15:00 | 834.00 | 2024-12-16 11:15:00 | 835.15 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-12-16 09:30:00 | 836.60 | 2024-12-16 11:15:00 | 835.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-12-30 14:45:00 | 983.30 | 2024-12-30 15:15:00 | 961.90 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-01 10:15:00 | 958.00 | 2025-01-01 15:15:00 | 971.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-01-13 14:15:00 | 894.25 | 2025-01-16 10:15:00 | 912.70 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-01-14 10:15:00 | 898.10 | 2025-01-16 10:15:00 | 912.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-14 12:00:00 | 899.30 | 2025-01-16 10:15:00 | 912.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-01-14 12:45:00 | 898.70 | 2025-01-16 10:15:00 | 912.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-02-01 11:45:00 | 803.45 | 2025-02-01 13:15:00 | 823.65 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-02-03 09:30:00 | 798.60 | 2025-02-05 09:15:00 | 849.85 | STOP_HIT | 1.00 | -6.42% |
| SELL | retest2 | 2025-02-04 11:45:00 | 806.75 | 2025-02-05 09:15:00 | 849.85 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-02-04 13:15:00 | 807.10 | 2025-02-05 09:15:00 | 849.85 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest1 | 2025-02-14 09:15:00 | 729.30 | 2025-02-17 09:15:00 | 692.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-14 09:15:00 | 729.30 | 2025-02-17 14:15:00 | 704.95 | STOP_HIT | 0.50 | 3.34% |
| BUY | retest2 | 2025-02-21 12:00:00 | 719.85 | 2025-02-25 13:15:00 | 712.80 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-02-24 09:45:00 | 719.30 | 2025-02-25 13:15:00 | 712.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-02-28 11:45:00 | 700.40 | 2025-02-28 14:15:00 | 665.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 12:30:00 | 700.15 | 2025-02-28 14:15:00 | 665.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 11:45:00 | 700.40 | 2025-03-04 14:15:00 | 647.35 | STOP_HIT | 0.50 | 7.57% |
| SELL | retest2 | 2025-02-28 12:30:00 | 700.15 | 2025-03-04 14:15:00 | 647.35 | STOP_HIT | 0.50 | 7.54% |
| BUY | retest2 | 2025-03-07 14:45:00 | 676.70 | 2025-03-10 12:15:00 | 667.65 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-10 09:30:00 | 677.90 | 2025-03-10 12:15:00 | 667.65 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-03-10 11:15:00 | 674.75 | 2025-03-10 12:15:00 | 667.65 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-12 11:15:00 | 651.75 | 2025-03-18 15:15:00 | 654.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-03-12 11:45:00 | 651.70 | 2025-03-18 15:15:00 | 654.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-03-13 13:00:00 | 650.45 | 2025-03-18 15:15:00 | 654.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-03-13 14:15:00 | 651.15 | 2025-03-18 15:15:00 | 654.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-03-21 10:30:00 | 701.50 | 2025-03-27 11:15:00 | 710.85 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-03-21 11:30:00 | 703.00 | 2025-03-27 11:15:00 | 710.85 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-04-04 10:00:00 | 673.00 | 2025-04-07 09:15:00 | 605.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 09:15:00 | 778.10 | 2025-04-25 13:15:00 | 781.00 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-04-25 12:00:00 | 775.50 | 2025-04-25 13:15:00 | 781.00 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-05-02 09:45:00 | 790.95 | 2025-05-06 13:15:00 | 789.50 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-05-02 14:30:00 | 791.80 | 2025-05-06 13:15:00 | 789.50 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-05-05 09:15:00 | 801.10 | 2025-05-06 13:15:00 | 789.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-05-09 12:30:00 | 869.85 | 2025-05-12 12:15:00 | 940.67 | TARGET_HIT | 1.00 | 8.14% |
| BUY | retest2 | 2025-05-09 13:45:00 | 855.15 | 2025-05-14 10:15:00 | 956.84 | TARGET_HIT | 1.00 | 11.89% |
| BUY | retest2 | 2025-05-12 09:15:00 | 866.80 | 2025-05-14 10:15:00 | 953.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-06-12 12:30:00 | 1211.90 | 2025-06-12 13:15:00 | 1187.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-27 14:45:00 | 1134.30 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-06-30 11:45:00 | 1140.30 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-02 11:15:00 | 1139.00 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1041.00 | 2025-07-30 13:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1038.70 | 2025-08-01 14:15:00 | 986.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1038.70 | 2025-08-04 11:15:00 | 997.70 | STOP_HIT | 0.50 | 3.95% |
| BUY | retest2 | 2025-08-25 11:45:00 | 997.40 | 2025-08-28 15:15:00 | 965.50 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-08-26 14:30:00 | 988.50 | 2025-08-28 15:15:00 | 965.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-09-18 09:15:00 | 1052.00 | 2025-09-24 09:15:00 | 1052.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-10-06 09:15:00 | 979.00 | 2025-10-06 10:15:00 | 988.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-29 10:45:00 | 1043.80 | 2025-11-03 09:15:00 | 1148.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1054.35 | 2025-11-03 09:15:00 | 1159.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1050.30 | 2025-11-03 09:15:00 | 1155.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1045.55 | 2025-11-03 09:15:00 | 1150.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-24 15:15:00 | 1090.30 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1092.80 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-25 14:15:00 | 1092.70 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-26 09:45:00 | 1093.00 | 2025-11-26 11:15:00 | 1103.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1094.20 | 2025-12-04 10:15:00 | 1109.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-03 12:30:00 | 1090.20 | 2025-12-04 10:15:00 | 1109.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1028.90 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-12-11 13:30:00 | 1033.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-11 14:30:00 | 1029.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-12-12 09:15:00 | 1029.50 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1027.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1025.50 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-12-15 11:45:00 | 1027.10 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1025.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-12-16 13:00:00 | 1012.10 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-12-16 13:45:00 | 1015.90 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1013.30 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-12-29 11:15:00 | 994.70 | 2025-12-30 14:15:00 | 944.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:45:00 | 994.70 | 2025-12-30 14:15:00 | 944.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 12:15:00 | 993.10 | 2025-12-30 14:15:00 | 943.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 15:15:00 | 990.40 | 2025-12-30 14:15:00 | 940.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:15:00 | 994.70 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-29 11:45:00 | 994.70 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-29 12:15:00 | 993.10 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-12-29 15:15:00 | 990.40 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-01-01 10:30:00 | 962.45 | 2026-01-05 13:15:00 | 967.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-01 12:00:00 | 962.80 | 2026-01-05 13:15:00 | 967.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-19 09:15:00 | 904.35 | 2026-01-22 11:15:00 | 913.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 911.75 | 2026-01-22 11:15:00 | 913.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-10 11:00:00 | 767.80 | 2026-02-12 09:15:00 | 729.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:00:00 | 767.80 | 2026-02-13 09:15:00 | 691.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:15:00 | 697.00 | 2026-02-23 15:15:00 | 704.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-24 09:15:00 | 687.80 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-24 15:15:00 | 694.35 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-25 10:30:00 | 687.50 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-26 12:15:00 | 694.50 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-03-16 10:15:00 | 647.45 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-03-16 11:00:00 | 647.45 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-03-16 11:45:00 | 647.95 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-03-16 12:30:00 | 646.35 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-17 15:00:00 | 642.65 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2026-04-10 10:30:00 | 685.65 | 2026-04-13 09:15:00 | 646.20 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest2 | 2026-04-10 15:15:00 | 684.70 | 2026-04-13 09:15:00 | 646.20 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2026-05-05 14:15:00 | 755.00 | 2026-05-08 14:15:00 | 830.50 | TARGET_HIT | 1.00 | 10.00% |
