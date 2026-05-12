# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1009.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 23
- **Target hits / Stop hits / Partials:** 7 / 24 / 12
- **Avg / median % per leg:** 1.60% / -1.73%
- **Sum % (uncompounded):** 68.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.26% | -20.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.26% | -20.4% |
| SELL (all) | 34 | 20 | 58.8% | 7 | 15 | 12 | 2.62% | 89.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 20 | 58.8% | 7 | 15 | 12 | 2.62% | 89.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 20 | 46.5% | 7 | 24 | 12 | 1.60% | 68.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1049.05 | 1114.74 | 1114.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1040.75 | 1114.01 | 1114.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 989.80 | 989.16 | 1032.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 14:00:00 | 989.80 | 989.16 | 1032.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1031.00 | 984.36 | 1025.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 1031.00 | 984.36 | 1025.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1049.15 | 985.01 | 1026.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 1049.15 | 985.01 | 1026.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1018.40 | 992.48 | 1026.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 1014.80 | 992.48 | 1026.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 964.06 | 992.84 | 1025.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-11 14:15:00 | 993.80 | 991.56 | 1023.85 | SL hit (close>ema200) qty=0.50 sl=991.56 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1065.85 | 931.31 | 930.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1120.95 | 937.35 | 933.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 15:15:00 | 1170.00 | 1171.63 | 1122.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:15:00 | 1172.30 | 1171.63 | 1122.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1150.90 | 1172.75 | 1127.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 1108.90 | 1172.75 | 1127.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1123.30 | 1176.04 | 1134.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 1123.30 | 1176.04 | 1134.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1133.00 | 1175.61 | 1134.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1141.90 | 1175.21 | 1134.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 11:15:00 | 1121.00 | 1172.95 | 1134.74 | SL hit (close<static) qty=1.00 sl=1123.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1054.80 | 1112.91 | 1112.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1044.20 | 1112.22 | 1112.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1133.00 | 1086.52 | 1097.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1133.00 | 1086.52 | 1097.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1133.00 | 1086.52 | 1097.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1133.00 | 1086.52 | 1097.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1094.50 | 1086.60 | 1097.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:00:00 | 1083.20 | 1100.09 | 1102.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:30:00 | 1083.20 | 1099.99 | 1102.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 1081.30 | 1099.77 | 1102.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:45:00 | 1083.50 | 1099.38 | 1102.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1029.04 | 1093.30 | 1098.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1029.04 | 1093.30 | 1098.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1027.23 | 1093.30 | 1098.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1029.33 | 1093.30 | 1098.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1102.70 | 1089.19 | 1095.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1102.70 | 1089.19 | 1095.88 | SL hit (close>ema200) qty=0.50 sl=1089.19 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1120.00 | 1098.52 | 1098.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1131.00 | 1099.82 | 1099.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1095.50 | 1102.55 | 1100.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 12:15:00 | 1095.50 | 1102.55 | 1100.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1095.50 | 1102.55 | 1100.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:00:00 | 1095.50 | 1102.55 | 1100.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1106.60 | 1102.59 | 1100.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 1128.00 | 1102.89 | 1100.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:30:00 | 1124.70 | 1114.57 | 1107.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 1125.20 | 1116.35 | 1108.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 1126.80 | 1116.41 | 1108.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1107.90 | 1121.89 | 1113.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 1106.70 | 1121.89 | 1113.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 1110.50 | 1121.78 | 1113.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 1094.80 | 1121.78 | 1113.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1094.20 | 1121.24 | 1113.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1094.20 | 1121.24 | 1113.02 | SL hit (close<static) qty=1.00 sl=1095.60 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 1080.70 | 1106.38 | 1106.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 1066.20 | 1104.27 | 1105.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 1092.30 | 1092.26 | 1098.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 15:00:00 | 1092.30 | 1092.26 | 1098.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1095.00 | 1084.24 | 1093.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1095.00 | 1084.24 | 1093.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1079.00 | 1084.19 | 1093.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 1069.30 | 1083.68 | 1092.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:00:00 | 1069.10 | 1083.35 | 1091.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:30:00 | 1064.20 | 1082.95 | 1091.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1065.10 | 1082.39 | 1090.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1088.50 | 1074.25 | 1084.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1088.50 | 1074.25 | 1084.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1080.40 | 1074.31 | 1084.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 1092.00 | 1074.31 | 1084.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1080.10 | 1074.37 | 1084.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:45:00 | 1079.30 | 1074.37 | 1084.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1097.40 | 1074.61 | 1084.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1097.40 | 1074.61 | 1084.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1087.10 | 1074.74 | 1084.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 1081.80 | 1074.85 | 1084.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 1083.00 | 1074.85 | 1084.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1083.80 | 1075.17 | 1084.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 1029.61 | 1073.54 | 1082.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1027.71 | 1073.06 | 1082.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1028.85 | 1073.06 | 1082.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1015.83 | 1067.61 | 1078.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1015.64 | 1067.61 | 1078.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1010.99 | 1067.61 | 1078.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1011.84 | 1067.61 | 1078.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 974.70 | 1062.88 | 1075.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 1004.55 | 913.51 | 913.14 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-07 13:15:00 | 1014.80 | 2025-02-11 09:15:00 | 964.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:15:00 | 1014.80 | 2025-02-11 14:15:00 | 993.80 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2025-05-21 09:30:00 | 1015.60 | 2025-05-22 10:15:00 | 1036.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-05-21 10:00:00 | 1008.95 | 2025-05-22 10:15:00 | 1036.00 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1141.90 | 2025-08-05 11:15:00 | 1121.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-22 14:00:00 | 1083.20 | 2025-09-26 13:15:00 | 1029.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:30:00 | 1083.20 | 2025-09-26 13:15:00 | 1029.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:45:00 | 1081.30 | 2025-09-26 13:15:00 | 1027.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1083.50 | 2025-09-26 13:15:00 | 1029.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 14:00:00 | 1083.20 | 2025-10-01 13:15:00 | 1102.70 | STOP_HIT | 0.50 | -1.80% |
| SELL | retest2 | 2025-09-22 14:30:00 | 1083.20 | 2025-10-01 13:15:00 | 1102.70 | STOP_HIT | 0.50 | -1.80% |
| SELL | retest2 | 2025-09-23 09:45:00 | 1081.30 | 2025-10-01 13:15:00 | 1102.70 | STOP_HIT | 0.50 | -1.98% |
| SELL | retest2 | 2025-09-23 11:45:00 | 1083.50 | 2025-10-01 13:15:00 | 1102.70 | STOP_HIT | 0.50 | -1.77% |
| SELL | retest2 | 2025-10-07 10:15:00 | 1083.50 | 2025-10-10 11:15:00 | 1108.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-10-07 13:15:00 | 1088.90 | 2025-10-10 11:15:00 | 1108.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-08 11:30:00 | 1088.90 | 2025-10-10 11:15:00 | 1108.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-10-08 12:30:00 | 1090.20 | 2025-10-10 11:15:00 | 1108.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-10-14 09:30:00 | 1083.10 | 2025-10-21 13:15:00 | 1144.50 | STOP_HIT | 1.00 | -5.67% |
| SELL | retest2 | 2025-10-15 10:30:00 | 1082.20 | 2025-10-21 13:15:00 | 1144.50 | STOP_HIT | 1.00 | -5.76% |
| SELL | retest2 | 2025-10-16 09:30:00 | 1080.80 | 2025-10-21 13:15:00 | 1144.50 | STOP_HIT | 1.00 | -5.89% |
| SELL | retest2 | 2025-10-16 11:15:00 | 1082.10 | 2025-10-21 13:15:00 | 1144.50 | STOP_HIT | 1.00 | -5.77% |
| BUY | retest2 | 2025-11-03 10:45:00 | 1128.00 | 2025-11-20 10:15:00 | 1094.20 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-11-10 09:30:00 | 1124.70 | 2025-11-20 10:15:00 | 1094.20 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-11-11 13:45:00 | 1125.20 | 2025-11-20 10:15:00 | 1094.20 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-11-12 10:15:00 | 1126.80 | 2025-11-20 10:15:00 | 1094.20 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1106.30 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-01 14:15:00 | 1106.10 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-01 15:00:00 | 1107.20 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-02 09:30:00 | 1107.80 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-24 11:45:00 | 1069.30 | 2026-01-12 10:15:00 | 1029.61 | PARTIAL | 0.50 | 3.71% |
| SELL | retest2 | 2025-12-24 15:00:00 | 1069.10 | 2026-01-12 11:15:00 | 1027.71 | PARTIAL | 0.50 | 3.87% |
| SELL | retest2 | 2025-12-26 09:30:00 | 1064.20 | 2026-01-12 11:15:00 | 1028.85 | PARTIAL | 0.50 | 3.32% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1065.10 | 2026-01-19 09:15:00 | 1015.83 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2026-01-07 11:30:00 | 1081.80 | 2026-01-19 09:15:00 | 1015.64 | PARTIAL | 0.50 | 6.12% |
| SELL | retest2 | 2026-01-07 12:15:00 | 1083.00 | 2026-01-19 09:15:00 | 1010.99 | PARTIAL | 0.50 | 6.65% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1083.80 | 2026-01-19 09:15:00 | 1011.84 | PARTIAL | 0.50 | 6.64% |
| SELL | retest2 | 2025-12-24 11:45:00 | 1069.30 | 2026-01-20 09:15:00 | 974.70 | TARGET_HIT | 0.50 | 8.85% |
| SELL | retest2 | 2025-12-24 15:00:00 | 1069.10 | 2026-01-20 09:15:00 | 975.42 | TARGET_HIT | 0.50 | 8.76% |
| SELL | retest2 | 2025-12-26 09:30:00 | 1064.20 | 2026-01-20 13:15:00 | 962.37 | TARGET_HIT | 0.50 | 9.57% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1065.10 | 2026-01-20 13:15:00 | 962.19 | TARGET_HIT | 0.50 | 9.66% |
| SELL | retest2 | 2026-01-07 11:30:00 | 1081.80 | 2026-01-20 13:15:00 | 957.78 | TARGET_HIT | 0.50 | 11.46% |
| SELL | retest2 | 2026-01-07 12:15:00 | 1083.00 | 2026-01-20 13:15:00 | 958.59 | TARGET_HIT | 0.50 | 11.49% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1083.80 | 2026-01-20 13:15:00 | 973.62 | TARGET_HIT | 0.50 | 10.17% |
