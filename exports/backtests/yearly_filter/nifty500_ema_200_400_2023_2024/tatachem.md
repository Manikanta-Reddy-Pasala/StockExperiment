# Tata Chemicals Ltd. (TATACHEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 782.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 20
- **Target hits / Stop hits / Partials:** 2 / 20 / 1
- **Avg / median % per leg:** -0.88% / -1.69%
- **Sum % (uncompounded):** -20.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 1 | 17 | 0 | -1.72% | -31.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 1 | 17 | 0 | -1.72% | -31.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.13% | 10.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 2.13% | 10.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 3 | 13.0% | 2 | 20 | 1 | -0.88% | -20.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 940.90 | 1026.76 | 1026.98 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 1043.10 | 997.47 | 997.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 1053.85 | 1001.80 | 999.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1062.65 | 1072.99 | 1046.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 1054.10 | 1072.99 | 1046.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1056.25 | 1073.40 | 1048.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:45:00 | 1051.70 | 1073.40 | 1048.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1042.05 | 1072.19 | 1048.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1042.05 | 1072.19 | 1048.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1033.15 | 1071.81 | 1048.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 1032.55 | 1071.81 | 1048.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-02-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 12:15:00 | 975.00 | 1034.04 | 1034.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 962.95 | 1031.52 | 1032.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 992.25 | 986.47 | 1003.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-04 14:00:00 | 992.25 | 986.47 | 1003.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 999.70 | 986.61 | 1003.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 1000.80 | 986.61 | 1003.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 1007.00 | 986.81 | 1003.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:15:00 | 1031.50 | 986.81 | 1003.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1077.80 | 987.71 | 1003.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:00:00 | 1077.80 | 987.71 | 1003.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 14:15:00 | 1318.05 | 1018.99 | 1018.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 14:15:00 | 1002.45 | 1076.28 | 1076.54 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 1124.25 | 1076.35 | 1076.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 09:15:00 | 1138.35 | 1086.60 | 1081.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 15:15:00 | 1090.00 | 1094.51 | 1086.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 09:15:00 | 1085.35 | 1094.51 | 1086.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1095.20 | 1094.52 | 1086.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 09:30:00 | 1099.30 | 1094.08 | 1087.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 14:15:00 | 1102.65 | 1094.07 | 1087.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1103.60 | 1095.59 | 1088.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:00:00 | 1104.20 | 1095.68 | 1088.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1103.00 | 1096.77 | 1089.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:30:00 | 1097.15 | 1096.77 | 1089.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1084.90 | 1096.84 | 1089.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-08 11:15:00 | 1084.90 | 1096.84 | 1089.95 | SL hit (close<static) qty=1.00 sl=1085.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 1060.00 | 1084.58 | 1084.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1040.00 | 1084.14 | 1084.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 1090.40 | 1074.62 | 1079.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 1090.40 | 1074.62 | 1079.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1090.40 | 1074.62 | 1079.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:45:00 | 1093.35 | 1074.62 | 1079.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 1093.55 | 1074.80 | 1079.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:45:00 | 1094.05 | 1074.80 | 1079.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 11:15:00 | 1091.55 | 1082.97 | 1082.96 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1061.15 | 1082.77 | 1082.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 1049.00 | 1081.71 | 1082.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1066.60 | 1061.89 | 1070.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:00:00 | 1066.60 | 1061.89 | 1070.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1076.65 | 1062.03 | 1070.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 1079.50 | 1062.03 | 1070.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1078.95 | 1062.19 | 1070.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 10:30:00 | 1079.65 | 1062.19 | 1070.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 1071.25 | 1063.56 | 1070.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 1071.30 | 1063.56 | 1070.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 1072.80 | 1063.65 | 1070.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 1072.15 | 1063.65 | 1070.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1072.55 | 1063.74 | 1070.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 1073.60 | 1063.74 | 1070.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1071.20 | 1063.81 | 1070.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1077.25 | 1063.81 | 1070.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1073.80 | 1067.05 | 1071.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1073.80 | 1067.05 | 1071.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1072.00 | 1067.10 | 1071.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1085.75 | 1067.10 | 1071.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1084.95 | 1067.28 | 1071.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 1091.60 | 1067.28 | 1071.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1079.90 | 1072.99 | 1074.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 11:00:00 | 1079.90 | 1072.99 | 1074.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1059.85 | 1053.06 | 1061.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 1062.75 | 1053.06 | 1061.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1060.70 | 1053.13 | 1061.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 1060.70 | 1053.13 | 1061.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1059.05 | 1053.19 | 1061.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:15:00 | 1061.10 | 1053.19 | 1061.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1060.50 | 1053.26 | 1061.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 1057.55 | 1053.33 | 1061.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:45:00 | 1058.15 | 1053.39 | 1061.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1055.85 | 1053.39 | 1061.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1072.45 | 1053.36 | 1061.37 | SL hit (close>static) qty=1.00 sl=1065.85 alert=retest2 |

### Cycle 10 — BUY (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 12:15:00 | 1144.05 | 1067.96 | 1067.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1163.05 | 1075.77 | 1072.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 11:15:00 | 1088.60 | 1091.35 | 1081.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:30:00 | 1088.05 | 1091.35 | 1081.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 1075.80 | 1091.16 | 1081.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:00:00 | 1075.80 | 1091.16 | 1081.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1071.25 | 1090.97 | 1081.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 1071.25 | 1090.97 | 1081.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1074.90 | 1090.81 | 1081.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1047.50 | 1090.81 | 1081.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1088.45 | 1089.55 | 1081.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1141.05 | 1089.50 | 1081.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 1077.50 | 1102.98 | 1089.74 | SL hit (close<static) qty=1.00 sl=1077.85 alert=retest2 |

### Cycle 11 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 1037.90 | 1094.41 | 1094.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 1025.75 | 1093.73 | 1094.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 10:15:00 | 1092.30 | 1089.07 | 1091.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 10:15:00 | 1092.30 | 1089.07 | 1091.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1092.30 | 1089.07 | 1091.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:30:00 | 1100.00 | 1089.07 | 1091.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1082.75 | 1089.01 | 1091.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:15:00 | 1072.65 | 1088.89 | 1091.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:15:00 | 1019.02 | 1074.93 | 1083.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 965.39 | 1048.72 | 1067.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 905.20 | 865.01 | 864.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 908.95 | 867.01 | 865.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 904.10 | 905.85 | 890.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 904.30 | 905.85 | 890.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 906.65 | 920.39 | 905.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:15:00 | 910.40 | 920.39 | 905.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 910.40 | 920.29 | 905.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 912.00 | 920.29 | 905.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-29 14:15:00 | 1003.20 | 936.07 | 919.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 907.65 | 940.70 | 940.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 907.05 | 940.37 | 940.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 776.15 | 774.89 | 808.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:00:00 | 776.15 | 774.89 | 808.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 705.00 | 656.91 | 687.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 761.25 | 657.72 | 687.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 731.70 | 658.45 | 687.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:15:00 | 731.90 | 658.45 | 687.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 692.40 | 680.99 | 693.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:15:00 | 694.00 | 680.99 | 693.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 693.50 | 681.11 | 693.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 695.75 | 681.11 | 693.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 695.20 | 681.25 | 693.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 707.80 | 681.25 | 693.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 712.80 | 681.56 | 693.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 712.80 | 681.56 | 693.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 816.00 | 703.45 | 703.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 822.80 | 706.69 | 704.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-28 09:30:00 | 1099.30 | 2024-07-08 11:15:00 | 1084.90 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-06-28 14:15:00 | 1102.65 | 2024-07-08 11:15:00 | 1084.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-03 09:15:00 | 1103.60 | 2024-07-08 11:15:00 | 1084.90 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-07-03 10:00:00 | 1104.20 | 2024-07-08 11:15:00 | 1084.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-09-25 14:15:00 | 1057.55 | 2024-09-27 09:15:00 | 1072.45 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-25 14:45:00 | 1058.15 | 2024-09-27 09:15:00 | 1072.45 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-09-25 15:15:00 | 1055.85 | 2024-09-27 09:15:00 | 1072.45 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1141.05 | 2024-10-25 11:15:00 | 1077.50 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest2 | 2024-10-28 10:30:00 | 1097.40 | 2024-10-29 10:15:00 | 1075.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-10-28 12:00:00 | 1093.45 | 2024-10-29 10:15:00 | 1075.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-10-29 13:00:00 | 1093.60 | 2024-11-11 09:15:00 | 1082.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-10-30 12:30:00 | 1124.00 | 2024-11-11 09:15:00 | 1082.50 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1119.20 | 2024-11-11 09:15:00 | 1082.50 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-11-04 13:00:00 | 1122.50 | 2024-11-11 09:15:00 | 1082.50 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-11-04 13:30:00 | 1118.55 | 2024-11-12 13:15:00 | 1079.95 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-11-11 14:30:00 | 1104.70 | 2024-11-12 13:15:00 | 1079.95 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-11-12 09:45:00 | 1104.80 | 2024-11-12 14:15:00 | 1072.55 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-11-25 11:15:00 | 1102.15 | 2024-12-13 09:15:00 | 1077.85 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-11-27 12:00:00 | 1103.50 | 2024-12-13 09:15:00 | 1077.85 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-12-24 13:15:00 | 1072.65 | 2025-01-02 10:15:00 | 1019.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 13:15:00 | 1072.65 | 2025-01-13 09:15:00 | 965.39 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-14 09:30:00 | 912.00 | 2025-07-29 14:15:00 | 1003.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 13:45:00 | 914.65 | 2025-10-10 11:15:00 | 907.65 | STOP_HIT | 1.00 | -0.77% |
