# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1475.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 240 |
| ALERT1 | 165 |
| ALERT2 | 161 |
| ALERT2_SKIP | 89 |
| ALERT3 | 449 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 162 |
| PARTIAL | 2 |
| TARGET_HIT | 5 |
| STOP_HIT | 160 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 165 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 107
- **Target hits / Stop hits / Partials:** 5 / 158 / 2
- **Avg / median % per leg:** 0.05% / -0.62%
- **Sum % (uncompounded):** 7.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 27 | 32.1% | 5 | 79 | 0 | 0.30% | 25.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 84 | 27 | 32.1% | 5 | 79 | 0 | 0.30% | 25.4% |
| SELL (all) | 81 | 31 | 38.3% | 0 | 79 | 2 | -0.22% | -17.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.71% | -0.7% |
| SELL @ 3rd Alert (retest2) | 80 | 31 | 38.8% | 0 | 78 | 2 | -0.21% | -16.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.71% | -0.7% |
| retest2 (combined) | 164 | 58 | 35.4% | 5 | 157 | 2 | 0.05% | 8.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-12 12:15:00 | 1098.39 | 1100.06 | 1100.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-12 13:15:00 | 1093.35 | 1098.72 | 1099.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-15 09:15:00 | 1099.49 | 1096.15 | 1097.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-15 09:15:00 | 1099.49 | 1096.15 | 1097.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 09:15:00 | 1099.49 | 1096.15 | 1097.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-15 10:00:00 | 1099.49 | 1096.15 | 1097.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-15 10:15:00 | 1093.68 | 1095.65 | 1097.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-15 11:15:00 | 1091.25 | 1095.65 | 1097.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-16 11:00:00 | 1092.49 | 1092.67 | 1094.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-16 11:30:00 | 1090.20 | 1091.83 | 1094.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 10:15:00 | 1092.39 | 1090.58 | 1092.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 1089.15 | 1090.29 | 1092.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-17 11:15:00 | 1088.20 | 1090.29 | 1092.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-18 10:15:00 | 1094.11 | 1089.53 | 1090.35 | SL hit (close>static) qty=1.00 sl=1093.95 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 11:15:00 | 1076.38 | 1074.11 | 1074.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 14:15:00 | 1078.85 | 1075.39 | 1074.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 09:15:00 | 1079.29 | 1081.26 | 1078.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 09:15:00 | 1079.29 | 1081.26 | 1078.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 1079.29 | 1081.26 | 1078.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 09:30:00 | 1078.13 | 1081.26 | 1078.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 1076.90 | 1080.38 | 1078.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:45:00 | 1077.01 | 1080.38 | 1078.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 1079.10 | 1080.13 | 1078.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:30:00 | 1076.75 | 1080.13 | 1078.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 1078.75 | 1079.85 | 1078.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:00:00 | 1078.75 | 1079.85 | 1078.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 13:15:00 | 1079.99 | 1079.88 | 1078.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 13:45:00 | 1078.76 | 1079.88 | 1078.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 14:15:00 | 1078.76 | 1079.66 | 1078.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 15:00:00 | 1078.76 | 1079.66 | 1078.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 1078.75 | 1079.47 | 1078.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 09:15:00 | 1085.44 | 1079.47 | 1078.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 1088.05 | 1081.19 | 1079.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:15:00 | 1089.20 | 1081.19 | 1079.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-31 11:15:00 | 1076.26 | 1080.57 | 1079.70 | SL hit (close<static) qty=1.00 sl=1076.72 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 11:15:00 | 1084.11 | 1091.48 | 1092.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 12:15:00 | 1081.80 | 1089.54 | 1091.14 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 1107.95 | 1091.96 | 1091.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 11:15:00 | 1118.75 | 1099.73 | 1095.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 10:15:00 | 1107.96 | 1110.27 | 1103.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-08 11:00:00 | 1107.96 | 1110.27 | 1103.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 1107.38 | 1108.26 | 1105.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 1109.40 | 1108.26 | 1105.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 10:15:00 | 1103.96 | 1107.44 | 1105.30 | SL hit (close<static) qty=1.00 sl=1104.37 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 1139.80 | 1142.66 | 1142.77 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 11:15:00 | 1148.51 | 1143.83 | 1143.29 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 13:15:00 | 1141.60 | 1143.93 | 1143.99 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 14:15:00 | 1144.60 | 1144.06 | 1144.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 15:15:00 | 1146.00 | 1144.45 | 1144.22 | Break + close above crossover candle high |

### Cycle 9 — SELL (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 09:15:00 | 1139.17 | 1143.39 | 1143.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 10:15:00 | 1133.78 | 1141.47 | 1142.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 09:15:00 | 1130.44 | 1128.26 | 1131.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 09:15:00 | 1130.44 | 1128.26 | 1131.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 1130.44 | 1128.26 | 1131.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 15:15:00 | 1125.25 | 1128.47 | 1130.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 09:15:00 | 1135.99 | 1129.46 | 1130.63 | SL hit (close>static) qty=1.00 sl=1135.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 1133.49 | 1131.56 | 1131.40 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 13:15:00 | 1128.83 | 1131.01 | 1131.17 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 14:15:00 | 1132.47 | 1131.30 | 1131.29 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 1127.50 | 1130.54 | 1130.94 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 1134.74 | 1131.36 | 1131.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 11:15:00 | 1136.07 | 1132.30 | 1131.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 12:15:00 | 1132.27 | 1132.29 | 1131.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 12:15:00 | 1132.27 | 1132.29 | 1131.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 12:15:00 | 1132.27 | 1132.29 | 1131.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 12:30:00 | 1132.18 | 1132.29 | 1131.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 1139.36 | 1133.71 | 1132.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 14:15:00 | 1141.18 | 1137.50 | 1135.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 11:15:00 | 1129.62 | 1137.68 | 1136.71 | SL hit (close<static) qty=1.00 sl=1132.27 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 14:15:00 | 1132.00 | 1135.46 | 1135.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 12:15:00 | 1126.76 | 1132.25 | 1134.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 14:15:00 | 1132.41 | 1131.93 | 1133.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-04 15:00:00 | 1132.41 | 1131.93 | 1133.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 1129.30 | 1131.40 | 1133.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 1136.07 | 1131.40 | 1133.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1137.30 | 1132.58 | 1133.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:00:00 | 1137.30 | 1132.58 | 1133.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 1143.17 | 1134.70 | 1134.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 1164.80 | 1144.10 | 1139.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 1146.89 | 1149.57 | 1145.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 1146.89 | 1149.57 | 1145.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 1146.89 | 1149.57 | 1145.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:30:00 | 1144.85 | 1149.57 | 1145.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 1144.92 | 1148.64 | 1145.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:00:00 | 1144.92 | 1148.64 | 1145.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 1150.55 | 1149.02 | 1145.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:30:00 | 1150.93 | 1148.11 | 1146.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 15:15:00 | 1143.45 | 1147.18 | 1145.76 | SL hit (close<static) qty=1.00 sl=1144.50 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 1140.76 | 1144.49 | 1144.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 11:15:00 | 1128.46 | 1141.28 | 1143.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1140.03 | 1134.98 | 1138.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1140.03 | 1134.98 | 1138.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1140.03 | 1134.98 | 1138.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 1140.03 | 1134.98 | 1138.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 1140.50 | 1136.08 | 1138.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 11:00:00 | 1140.50 | 1136.08 | 1138.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 1140.43 | 1136.95 | 1139.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:00:00 | 1140.43 | 1136.95 | 1139.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 1137.03 | 1136.97 | 1138.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 13:15:00 | 1136.22 | 1136.97 | 1138.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-11 14:15:00 | 1141.37 | 1137.70 | 1138.83 | SL hit (close>static) qty=1.00 sl=1140.50 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 1148.53 | 1140.88 | 1140.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 14:15:00 | 1153.50 | 1146.85 | 1143.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 11:15:00 | 1146.20 | 1149.72 | 1146.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 11:15:00 | 1146.20 | 1149.72 | 1146.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 1146.20 | 1149.72 | 1146.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 11:45:00 | 1145.75 | 1149.72 | 1146.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 12:15:00 | 1145.48 | 1148.87 | 1146.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:45:00 | 1144.56 | 1148.87 | 1146.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 1144.24 | 1147.94 | 1146.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 1144.24 | 1147.94 | 1146.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 1140.42 | 1146.44 | 1145.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:45:00 | 1139.52 | 1146.44 | 1145.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 1142.50 | 1144.79 | 1144.94 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 1148.74 | 1145.21 | 1145.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 1156.36 | 1148.12 | 1146.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 12:15:00 | 1151.88 | 1154.06 | 1150.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 12:15:00 | 1151.88 | 1154.06 | 1150.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 1151.88 | 1154.06 | 1150.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:15:00 | 1148.80 | 1154.06 | 1150.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 1149.29 | 1153.11 | 1150.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 13:30:00 | 1149.29 | 1153.11 | 1150.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 1154.33 | 1153.35 | 1150.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:30:00 | 1149.30 | 1153.35 | 1150.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 1149.25 | 1153.05 | 1151.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:00:00 | 1149.25 | 1153.05 | 1151.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 1146.50 | 1151.74 | 1150.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:30:00 | 1146.99 | 1151.74 | 1150.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 1145.39 | 1149.78 | 1150.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 09:15:00 | 1139.17 | 1145.10 | 1147.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 11:15:00 | 1150.35 | 1145.43 | 1146.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 11:15:00 | 1150.35 | 1145.43 | 1146.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 1150.35 | 1145.43 | 1146.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:00:00 | 1150.35 | 1145.43 | 1146.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 1150.49 | 1146.44 | 1147.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:45:00 | 1152.32 | 1146.44 | 1147.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 1150.05 | 1147.07 | 1147.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 15:00:00 | 1150.05 | 1147.07 | 1147.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 1152.49 | 1148.15 | 1147.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 1156.48 | 1149.82 | 1148.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 11:15:00 | 1149.10 | 1150.65 | 1149.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 11:15:00 | 1149.10 | 1150.65 | 1149.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 1149.10 | 1150.65 | 1149.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:45:00 | 1148.47 | 1150.65 | 1149.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 1154.34 | 1151.39 | 1149.68 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 09:15:00 | 1142.98 | 1148.37 | 1148.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 11:15:00 | 1139.96 | 1145.78 | 1147.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 1140.30 | 1138.90 | 1141.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 1140.30 | 1138.90 | 1141.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 1140.30 | 1138.90 | 1141.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-26 11:30:00 | 1136.26 | 1138.99 | 1141.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 09:30:00 | 1137.86 | 1139.00 | 1140.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 10:45:00 | 1137.20 | 1135.10 | 1138.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 15:15:00 | 1132.50 | 1127.03 | 1126.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 15:15:00 | 1132.50 | 1127.03 | 1126.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 11:15:00 | 1139.81 | 1130.71 | 1128.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-03 10:15:00 | 1136.03 | 1136.67 | 1133.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-03 10:45:00 | 1138.21 | 1136.67 | 1133.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 1132.65 | 1135.87 | 1133.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 11:45:00 | 1132.74 | 1135.87 | 1133.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 12:15:00 | 1125.70 | 1133.83 | 1132.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 13:00:00 | 1125.70 | 1133.83 | 1132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 13:15:00 | 1125.49 | 1132.17 | 1131.73 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 14:15:00 | 1125.59 | 1130.85 | 1131.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 15:15:00 | 1121.30 | 1128.94 | 1130.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 1130.18 | 1129.19 | 1130.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 1130.18 | 1129.19 | 1130.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1130.18 | 1129.19 | 1130.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:30:00 | 1131.75 | 1129.19 | 1130.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 1125.65 | 1128.48 | 1129.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:15:00 | 1120.75 | 1126.29 | 1128.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 13:45:00 | 1122.49 | 1118.82 | 1122.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 15:00:00 | 1122.37 | 1119.53 | 1122.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 10:15:00 | 1099.25 | 1090.54 | 1089.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 10:15:00 | 1099.25 | 1090.54 | 1089.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 11:15:00 | 1099.74 | 1092.38 | 1090.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 10:15:00 | 1097.81 | 1097.85 | 1094.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 11:00:00 | 1097.81 | 1097.85 | 1094.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 1104.45 | 1100.83 | 1097.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 09:45:00 | 1099.70 | 1100.83 | 1097.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1106.40 | 1110.39 | 1107.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:45:00 | 1106.86 | 1110.39 | 1107.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1107.78 | 1109.87 | 1107.14 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 11:15:00 | 1103.99 | 1106.67 | 1106.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 14:15:00 | 1101.53 | 1104.93 | 1105.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 12:15:00 | 1103.64 | 1102.61 | 1104.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 13:00:00 | 1103.64 | 1102.61 | 1104.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 1102.59 | 1102.61 | 1104.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 1101.00 | 1102.66 | 1103.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 09:15:00 | 1105.45 | 1103.37 | 1104.01 | SL hit (close>static) qty=1.00 sl=1104.25 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 11:15:00 | 1111.00 | 1105.16 | 1104.73 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 10:15:00 | 1095.20 | 1104.50 | 1104.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 11:15:00 | 1091.98 | 1101.99 | 1103.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 14:15:00 | 1100.11 | 1097.93 | 1101.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 14:15:00 | 1100.11 | 1097.93 | 1101.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 1100.11 | 1097.93 | 1101.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 1100.11 | 1097.93 | 1101.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 09:15:00 | 1097.45 | 1098.04 | 1100.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 09:30:00 | 1104.88 | 1098.04 | 1100.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 1087.55 | 1094.50 | 1097.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 10:15:00 | 1086.37 | 1094.50 | 1097.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-04 15:15:00 | 1085.01 | 1089.03 | 1093.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 11:15:00 | 1099.26 | 1091.98 | 1093.21 | SL hit (close>static) qty=1.00 sl=1098.99 alert=retest2 |

### Cycle 30 — BUY (started 2023-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 14:15:00 | 1097.34 | 1094.04 | 1093.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 15:15:00 | 1098.95 | 1095.02 | 1094.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 1096.46 | 1097.98 | 1096.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 13:15:00 | 1096.46 | 1097.98 | 1096.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 1096.46 | 1097.98 | 1096.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 14:00:00 | 1096.46 | 1097.98 | 1096.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 1100.00 | 1098.38 | 1096.72 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 14:15:00 | 1094.68 | 1096.23 | 1096.25 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 1105.00 | 1096.89 | 1096.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 14:15:00 | 1106.10 | 1100.84 | 1098.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 11:15:00 | 1113.32 | 1115.21 | 1109.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:00:00 | 1113.32 | 1115.21 | 1109.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 13:15:00 | 1112.18 | 1114.02 | 1110.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 13:30:00 | 1112.67 | 1114.02 | 1110.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 1108.72 | 1112.96 | 1109.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 1108.72 | 1112.96 | 1109.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 1109.43 | 1112.26 | 1109.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 09:15:00 | 1113.89 | 1112.26 | 1109.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 11:00:00 | 1113.12 | 1112.87 | 1110.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 10:15:00 | 1124.60 | 1129.64 | 1129.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 10:15:00 | 1124.60 | 1129.64 | 1129.76 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 13:15:00 | 1131.06 | 1127.73 | 1127.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 1136.95 | 1131.23 | 1129.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 1140.00 | 1142.25 | 1136.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 10:00:00 | 1140.00 | 1142.25 | 1136.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 1142.00 | 1142.20 | 1137.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 1137.24 | 1142.20 | 1137.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 13:15:00 | 1140.85 | 1142.08 | 1138.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:30:00 | 1143.76 | 1142.08 | 1138.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 1134.25 | 1139.99 | 1138.62 | SL hit (close<static) qty=1.00 sl=1138.29 alert=retest2 |

### Cycle 35 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 1129.05 | 1136.56 | 1137.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 1127.41 | 1134.73 | 1136.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 12:15:00 | 1135.00 | 1130.97 | 1133.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 12:15:00 | 1135.00 | 1130.97 | 1133.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 1135.00 | 1130.97 | 1133.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:00:00 | 1135.00 | 1130.97 | 1133.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 1133.38 | 1131.45 | 1133.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 15:00:00 | 1124.24 | 1130.01 | 1132.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 09:30:00 | 1123.68 | 1126.49 | 1130.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 10:15:00 | 1135.63 | 1124.76 | 1126.28 | SL hit (close>static) qty=1.00 sl=1135.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 12:15:00 | 1137.89 | 1128.14 | 1127.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 13:15:00 | 1144.20 | 1131.35 | 1129.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 15:15:00 | 1145.00 | 1145.50 | 1140.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:15:00 | 1147.99 | 1145.50 | 1140.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1143.84 | 1145.98 | 1143.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 15:15:00 | 1150.01 | 1145.23 | 1144.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 14:15:00 | 1155.48 | 1158.49 | 1158.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 14:15:00 | 1155.48 | 1158.49 | 1158.79 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 10:15:00 | 1163.00 | 1159.71 | 1159.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 14:15:00 | 1166.75 | 1161.63 | 1160.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 13:15:00 | 1163.46 | 1163.99 | 1162.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 14:00:00 | 1163.46 | 1163.99 | 1162.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1163.67 | 1163.92 | 1162.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 14:30:00 | 1162.92 | 1163.92 | 1162.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 1162.76 | 1163.69 | 1162.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:15:00 | 1158.67 | 1163.69 | 1162.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 1158.44 | 1162.64 | 1162.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:45:00 | 1157.95 | 1162.64 | 1162.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 1166.13 | 1163.34 | 1162.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 10:45:00 | 1164.25 | 1163.34 | 1162.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 1212.00 | 1216.88 | 1211.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 15:00:00 | 1212.00 | 1216.88 | 1211.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 1212.26 | 1215.95 | 1211.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:15:00 | 1200.06 | 1215.95 | 1211.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 1193.80 | 1211.52 | 1209.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 10:00:00 | 1193.80 | 1211.52 | 1209.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-10-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 10:15:00 | 1184.15 | 1206.05 | 1207.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 13:15:00 | 1180.40 | 1194.48 | 1201.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 10:15:00 | 1194.70 | 1189.23 | 1195.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 10:15:00 | 1194.70 | 1189.23 | 1195.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1194.70 | 1189.23 | 1195.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 1194.70 | 1189.23 | 1195.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 1199.86 | 1191.36 | 1196.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 1199.86 | 1191.36 | 1196.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 1203.50 | 1193.79 | 1196.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:45:00 | 1204.51 | 1193.79 | 1196.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 13:15:00 | 1206.26 | 1196.28 | 1197.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 13:45:00 | 1210.26 | 1196.28 | 1197.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 1202.50 | 1198.84 | 1198.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 12:15:00 | 1209.36 | 1202.12 | 1200.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 1198.50 | 1207.81 | 1205.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 1198.50 | 1207.81 | 1205.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 1198.50 | 1207.81 | 1205.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 1198.50 | 1207.81 | 1205.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 1196.70 | 1205.59 | 1205.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 1196.70 | 1205.59 | 1205.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 11:15:00 | 1195.00 | 1203.47 | 1204.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 1192.12 | 1201.20 | 1203.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 11:15:00 | 1199.99 | 1196.75 | 1199.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 1199.99 | 1196.75 | 1199.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 1199.99 | 1196.75 | 1199.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:00:00 | 1199.99 | 1196.75 | 1199.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 1201.23 | 1197.65 | 1199.63 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 15:15:00 | 1204.11 | 1201.19 | 1200.95 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 15:15:00 | 1200.20 | 1201.33 | 1201.37 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 09:15:00 | 1208.14 | 1202.70 | 1201.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 12:15:00 | 1214.99 | 1207.31 | 1204.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 09:15:00 | 1214.95 | 1215.05 | 1211.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 1214.95 | 1215.05 | 1211.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1214.95 | 1215.05 | 1211.73 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 1207.04 | 1211.60 | 1212.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-13 15:15:00 | 1201.61 | 1205.03 | 1207.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 11:15:00 | 1205.56 | 1204.82 | 1206.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 11:15:00 | 1205.56 | 1204.82 | 1206.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 1205.56 | 1204.82 | 1206.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 12:00:00 | 1205.56 | 1204.82 | 1206.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 12:15:00 | 1206.90 | 1205.24 | 1206.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 13:15:00 | 1209.49 | 1205.24 | 1206.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 13:15:00 | 1208.71 | 1205.93 | 1206.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 15:15:00 | 1206.00 | 1206.74 | 1207.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 09:45:00 | 1206.10 | 1206.79 | 1207.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 10:15:00 | 1212.50 | 1207.93 | 1207.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 10:15:00 | 1212.50 | 1207.93 | 1207.52 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-11-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 14:15:00 | 1203.74 | 1207.18 | 1207.35 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 1220.08 | 1209.13 | 1208.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 09:15:00 | 1227.24 | 1218.63 | 1216.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 13:15:00 | 1220.26 | 1220.67 | 1218.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-23 14:00:00 | 1220.26 | 1220.67 | 1218.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 14:15:00 | 1219.43 | 1220.43 | 1218.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-23 15:00:00 | 1219.43 | 1220.43 | 1218.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 15:15:00 | 1220.74 | 1220.49 | 1218.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:15:00 | 1218.67 | 1220.49 | 1218.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1214.70 | 1219.33 | 1218.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:45:00 | 1216.88 | 1219.33 | 1218.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1211.72 | 1217.81 | 1217.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:00:00 | 1211.72 | 1217.81 | 1217.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 1211.59 | 1216.56 | 1217.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 1206.49 | 1214.55 | 1216.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 1211.22 | 1204.46 | 1208.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 1211.22 | 1204.46 | 1208.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1211.22 | 1204.46 | 1208.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 1211.22 | 1204.46 | 1208.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1211.10 | 1205.79 | 1208.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 1209.64 | 1205.79 | 1208.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 1210.95 | 1207.62 | 1209.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:45:00 | 1212.95 | 1207.62 | 1209.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 1205.15 | 1207.12 | 1208.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:30:00 | 1203.50 | 1206.33 | 1208.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 11:00:00 | 1201.76 | 1203.38 | 1205.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 14:00:00 | 1200.25 | 1202.44 | 1204.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 14:15:00 | 1213.51 | 1204.65 | 1205.63 | SL hit (close>static) qty=1.00 sl=1211.73 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 09:15:00 | 1211.04 | 1206.78 | 1206.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 14:15:00 | 1217.49 | 1210.62 | 1208.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 12:15:00 | 1214.48 | 1214.70 | 1211.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-04 12:30:00 | 1214.75 | 1214.70 | 1211.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 13:15:00 | 1217.89 | 1215.34 | 1212.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 14:15:00 | 1222.28 | 1215.34 | 1212.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 15:15:00 | 1222.50 | 1215.90 | 1212.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 10:15:00 | 1246.05 | 1248.43 | 1248.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 1246.05 | 1248.43 | 1248.44 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 1252.39 | 1248.29 | 1248.16 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 09:15:00 | 1244.94 | 1248.09 | 1248.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-14 11:15:00 | 1241.82 | 1246.21 | 1247.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-19 09:15:00 | 1249.90 | 1224.77 | 1226.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 1249.90 | 1224.77 | 1226.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1249.90 | 1224.77 | 1226.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 1249.90 | 1224.77 | 1226.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 10:15:00 | 1277.40 | 1235.30 | 1231.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 1277.45 | 1268.12 | 1262.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 1352.50 | 1353.44 | 1335.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 11:00:00 | 1352.50 | 1353.44 | 1335.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 1343.75 | 1351.54 | 1345.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 12:45:00 | 1343.36 | 1351.54 | 1345.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 1340.90 | 1349.41 | 1345.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:45:00 | 1340.51 | 1349.41 | 1345.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 1331.01 | 1342.97 | 1343.07 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 13:15:00 | 1349.54 | 1343.66 | 1343.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 14:15:00 | 1355.23 | 1345.97 | 1344.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 09:15:00 | 1330.25 | 1344.67 | 1344.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 09:15:00 | 1330.25 | 1344.67 | 1344.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1330.25 | 1344.67 | 1344.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:45:00 | 1330.48 | 1344.67 | 1344.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 10:15:00 | 1335.75 | 1342.89 | 1343.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 1323.88 | 1334.70 | 1338.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 15:15:00 | 1277.50 | 1275.22 | 1282.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-15 09:15:00 | 1273.40 | 1275.22 | 1282.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1283.00 | 1276.78 | 1282.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 1283.00 | 1276.78 | 1282.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 1279.63 | 1277.35 | 1282.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 11:30:00 | 1277.88 | 1277.65 | 1281.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 12:45:00 | 1277.83 | 1276.65 | 1281.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 15:15:00 | 1258.25 | 1246.00 | 1244.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 15:15:00 | 1258.25 | 1246.00 | 1244.66 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 14:15:00 | 1241.47 | 1244.20 | 1244.50 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 1247.63 | 1244.70 | 1244.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 14:15:00 | 1249.88 | 1246.35 | 1245.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 1245.15 | 1253.24 | 1250.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 1245.15 | 1253.24 | 1250.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1245.15 | 1253.24 | 1250.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 1245.15 | 1253.24 | 1250.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 1244.50 | 1251.49 | 1249.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 1250.50 | 1251.49 | 1249.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 1245.00 | 1250.08 | 1249.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:45:00 | 1245.75 | 1250.08 | 1249.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 1249.83 | 1250.03 | 1249.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:15:00 | 1242.93 | 1250.03 | 1249.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 12:15:00 | 1245.03 | 1249.03 | 1249.05 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 1254.35 | 1249.42 | 1249.18 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 1244.03 | 1248.71 | 1249.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 14:15:00 | 1234.63 | 1244.02 | 1246.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 10:15:00 | 1243.65 | 1242.23 | 1244.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 10:15:00 | 1243.65 | 1242.23 | 1244.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1243.65 | 1242.23 | 1244.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:45:00 | 1243.78 | 1242.23 | 1244.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 1241.47 | 1242.08 | 1244.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:30:00 | 1244.43 | 1242.08 | 1244.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 1241.55 | 1240.17 | 1242.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 1244.45 | 1240.17 | 1242.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 1237.65 | 1239.67 | 1242.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 14:30:00 | 1234.33 | 1238.63 | 1240.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 10:15:00 | 1226.30 | 1230.51 | 1233.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 1233.93 | 1231.81 | 1234.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 13:15:00 | 1247.65 | 1236.51 | 1235.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 13:15:00 | 1247.65 | 1236.51 | 1235.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 15:15:00 | 1251.50 | 1241.49 | 1238.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 1228.60 | 1238.91 | 1237.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 1228.60 | 1238.91 | 1237.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1228.60 | 1238.91 | 1237.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 1228.60 | 1238.91 | 1237.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2024-02-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 10:15:00 | 1218.35 | 1234.80 | 1235.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 12:15:00 | 1214.50 | 1228.20 | 1232.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 11:15:00 | 1219.45 | 1218.45 | 1224.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-09 11:30:00 | 1222.13 | 1218.45 | 1224.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 1228.05 | 1220.70 | 1224.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 13:30:00 | 1229.53 | 1220.70 | 1224.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 1225.70 | 1221.70 | 1224.54 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 13:15:00 | 1230.20 | 1226.42 | 1225.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 09:15:00 | 1234.38 | 1228.91 | 1227.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 12:15:00 | 1229.00 | 1230.83 | 1228.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-13 13:00:00 | 1229.00 | 1230.83 | 1228.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 1229.10 | 1230.49 | 1228.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:45:00 | 1227.03 | 1230.49 | 1228.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1231.53 | 1230.70 | 1229.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:45:00 | 1228.78 | 1230.70 | 1229.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 1228.00 | 1230.16 | 1228.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:30:00 | 1221.38 | 1228.44 | 1228.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 1218.50 | 1226.45 | 1227.37 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 1243.30 | 1228.52 | 1227.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 1245.03 | 1231.82 | 1229.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 10:15:00 | 1229.93 | 1231.86 | 1229.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 1229.93 | 1231.86 | 1229.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 1229.93 | 1231.86 | 1229.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 1229.60 | 1231.86 | 1229.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 1227.00 | 1230.89 | 1229.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 11:45:00 | 1225.00 | 1230.89 | 1229.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 1224.97 | 1229.71 | 1229.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 13:30:00 | 1228.50 | 1229.49 | 1229.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 15:00:00 | 1228.88 | 1229.37 | 1229.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 15:15:00 | 1226.50 | 1228.79 | 1228.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-02-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 15:15:00 | 1226.50 | 1228.79 | 1228.87 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1234.00 | 1229.84 | 1229.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 12:15:00 | 1241.50 | 1234.77 | 1231.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1249.95 | 1253.26 | 1246.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 10:00:00 | 1249.95 | 1253.26 | 1246.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1267.53 | 1268.51 | 1262.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 1267.53 | 1268.51 | 1262.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1289.47 | 1288.56 | 1283.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 14:30:00 | 1295.25 | 1291.47 | 1286.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:30:00 | 1296.90 | 1293.10 | 1288.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 10:30:00 | 1296.00 | 1292.79 | 1288.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 15:15:00 | 1280.13 | 1287.93 | 1287.86 | SL hit (close<static) qty=1.00 sl=1283.38 alert=retest2 |

### Cycle 71 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 1287.05 | 1287.75 | 1287.78 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 10:15:00 | 1296.18 | 1289.44 | 1288.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 11:15:00 | 1297.95 | 1291.14 | 1289.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 12:15:00 | 1290.58 | 1291.03 | 1289.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 12:15:00 | 1290.58 | 1291.03 | 1289.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1290.58 | 1291.03 | 1289.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:00:00 | 1290.58 | 1291.03 | 1289.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 1291.75 | 1291.17 | 1289.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:45:00 | 1288.78 | 1291.17 | 1289.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 1298.75 | 1292.69 | 1290.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-29 14:30:00 | 1296.63 | 1292.69 | 1290.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1299.50 | 1294.05 | 1291.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:45:00 | 1306.72 | 1296.74 | 1292.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 13:45:00 | 1303.33 | 1300.14 | 1295.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 15:15:00 | 1290.00 | 1295.46 | 1295.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 15:15:00 | 1290.00 | 1295.46 | 1295.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 10:15:00 | 1282.68 | 1292.38 | 1294.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 1267.50 | 1267.45 | 1274.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 11:15:00 | 1273.55 | 1269.58 | 1274.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 11:15:00 | 1273.55 | 1269.58 | 1274.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:30:00 | 1271.93 | 1269.58 | 1274.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 1279.93 | 1271.65 | 1275.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 12:45:00 | 1279.97 | 1271.65 | 1275.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 1276.63 | 1272.65 | 1275.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:15:00 | 1280.68 | 1272.65 | 1275.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 1279.93 | 1274.10 | 1275.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:45:00 | 1279.97 | 1274.10 | 1275.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 1288.28 | 1277.64 | 1276.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 10:15:00 | 1311.58 | 1284.43 | 1280.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 09:15:00 | 1288.55 | 1297.02 | 1289.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 09:15:00 | 1288.55 | 1297.02 | 1289.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1288.55 | 1297.02 | 1289.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:00:00 | 1288.55 | 1297.02 | 1289.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1292.70 | 1296.16 | 1290.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 15:15:00 | 1297.50 | 1294.34 | 1291.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-13 13:15:00 | 1286.20 | 1296.90 | 1294.73 | SL hit (close<static) qty=1.00 sl=1286.65 alert=retest2 |

### Cycle 75 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 1287.55 | 1297.82 | 1297.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 1265.95 | 1287.02 | 1291.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 09:15:00 | 1263.40 | 1259.19 | 1271.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 09:45:00 | 1263.00 | 1259.19 | 1271.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 11:15:00 | 1270.13 | 1262.59 | 1271.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 11:30:00 | 1273.28 | 1262.59 | 1271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1274.13 | 1264.90 | 1271.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 1274.13 | 1264.90 | 1271.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 1272.20 | 1266.36 | 1271.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:30:00 | 1274.97 | 1266.36 | 1271.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 1276.38 | 1268.36 | 1271.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:45:00 | 1272.13 | 1268.36 | 1271.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 15:15:00 | 1275.05 | 1269.70 | 1272.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 09:15:00 | 1269.43 | 1269.70 | 1272.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1272.47 | 1271.20 | 1272.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:30:00 | 1274.58 | 1271.20 | 1272.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 1266.28 | 1270.21 | 1271.96 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 1280.68 | 1274.17 | 1273.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 1291.78 | 1281.43 | 1277.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 1289.13 | 1290.20 | 1284.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 10:15:00 | 1286.83 | 1290.20 | 1284.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 1285.30 | 1289.22 | 1284.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 11:00:00 | 1285.30 | 1289.22 | 1284.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 11:15:00 | 1280.83 | 1287.54 | 1284.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:00:00 | 1280.83 | 1287.54 | 1284.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 12:15:00 | 1277.68 | 1285.57 | 1283.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 12:45:00 | 1277.60 | 1285.57 | 1283.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1285.00 | 1284.38 | 1283.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 1293.93 | 1284.38 | 1283.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 09:15:00 | 1278.25 | 1300.66 | 1299.95 | SL hit (close<static) qty=1.00 sl=1280.63 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 10:15:00 | 1273.68 | 1295.26 | 1297.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 1266.63 | 1280.34 | 1288.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 15:15:00 | 1275.50 | 1274.34 | 1280.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-05 09:15:00 | 1279.25 | 1274.34 | 1280.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1279.10 | 1275.29 | 1280.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:30:00 | 1277.25 | 1274.68 | 1279.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-12 09:15:00 | 1276.78 | 1262.36 | 1260.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 1276.78 | 1262.36 | 1260.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-15 10:15:00 | 1281.00 | 1270.66 | 1266.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 14:15:00 | 1276.33 | 1276.53 | 1271.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-15 15:00:00 | 1276.33 | 1276.53 | 1271.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1282.50 | 1278.05 | 1272.84 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 10:15:00 | 1230.50 | 1265.46 | 1269.83 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 12:15:00 | 1246.10 | 1235.69 | 1235.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 13:15:00 | 1250.90 | 1238.73 | 1236.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 1247.10 | 1249.85 | 1244.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 12:45:00 | 1248.58 | 1249.85 | 1244.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 1249.90 | 1249.80 | 1245.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:45:00 | 1250.72 | 1249.80 | 1245.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 1275.00 | 1259.22 | 1251.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:30:00 | 1254.00 | 1259.22 | 1251.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 1261.18 | 1266.81 | 1259.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:30:00 | 1259.50 | 1266.81 | 1259.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1256.75 | 1264.80 | 1259.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:00:00 | 1256.75 | 1264.80 | 1259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 1245.33 | 1260.91 | 1258.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:00:00 | 1245.33 | 1260.91 | 1258.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 1246.38 | 1258.00 | 1256.96 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 1240.83 | 1254.57 | 1255.50 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1260.05 | 1255.59 | 1255.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 09:15:00 | 1265.53 | 1259.02 | 1257.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 14:15:00 | 1255.47 | 1260.28 | 1259.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 14:15:00 | 1255.47 | 1260.28 | 1259.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 1255.47 | 1260.28 | 1259.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 15:00:00 | 1255.47 | 1260.28 | 1259.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 1255.50 | 1259.33 | 1258.74 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-05-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 09:15:00 | 1246.58 | 1256.78 | 1257.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 1241.33 | 1253.69 | 1256.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 1257.15 | 1235.64 | 1239.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 1257.15 | 1235.64 | 1239.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 1257.15 | 1235.64 | 1239.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:45:00 | 1251.63 | 1235.64 | 1239.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1256.43 | 1239.80 | 1240.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:45:00 | 1261.00 | 1239.80 | 1240.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 11:15:00 | 1256.78 | 1243.20 | 1242.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 11:15:00 | 1262.20 | 1253.55 | 1248.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1257.58 | 1260.50 | 1254.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 1257.58 | 1260.50 | 1254.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1257.58 | 1260.50 | 1254.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 1255.97 | 1260.50 | 1254.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1258.58 | 1260.12 | 1254.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 1270.38 | 1257.55 | 1255.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 11:15:00 | 1269.95 | 1261.08 | 1257.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 13:00:00 | 1266.60 | 1262.61 | 1258.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 14:45:00 | 1266.47 | 1263.94 | 1260.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1259.28 | 1262.94 | 1260.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:30:00 | 1259.15 | 1262.94 | 1260.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 1261.05 | 1262.57 | 1260.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 1258.80 | 1262.57 | 1260.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 1256.15 | 1261.28 | 1260.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 1256.15 | 1261.28 | 1260.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 12:15:00 | 1258.43 | 1260.71 | 1259.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 1242.97 | 1256.33 | 1258.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 1242.97 | 1256.33 | 1258.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 1238.05 | 1246.26 | 1251.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 1232.55 | 1225.80 | 1234.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 1232.55 | 1225.80 | 1234.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1232.55 | 1225.80 | 1234.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1232.55 | 1225.80 | 1234.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1233.43 | 1227.33 | 1233.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1223.28 | 1227.33 | 1233.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-18 09:15:00 | 1249.55 | 1229.10 | 1230.54 | SL hit (close>static) qty=1.00 sl=1237.33 alert=retest2 |

### Cycle 86 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 1253.75 | 1234.03 | 1232.65 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 1229.68 | 1232.53 | 1232.75 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 1239.47 | 1233.83 | 1233.30 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 1231.75 | 1233.97 | 1234.06 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 1239.58 | 1235.14 | 1234.58 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 1231.15 | 1234.74 | 1234.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 15:15:00 | 1228.00 | 1232.35 | 1233.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 1229.65 | 1229.28 | 1230.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1229.65 | 1229.28 | 1230.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1229.65 | 1229.28 | 1230.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1229.65 | 1229.28 | 1230.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1239.33 | 1231.29 | 1231.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:00:00 | 1239.33 | 1231.29 | 1231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 1238.50 | 1232.73 | 1232.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 12:15:00 | 1245.00 | 1235.19 | 1233.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 15:15:00 | 1234.22 | 1236.63 | 1234.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 15:15:00 | 1234.22 | 1236.63 | 1234.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1234.22 | 1236.63 | 1234.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1220.15 | 1236.63 | 1234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1206.22 | 1230.55 | 1232.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1204.45 | 1216.16 | 1223.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 09:15:00 | 1191.03 | 1184.37 | 1193.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1191.03 | 1184.37 | 1193.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1191.03 | 1184.37 | 1193.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:45:00 | 1191.43 | 1184.37 | 1193.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1173.33 | 1182.16 | 1191.28 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 15:15:00 | 1216.08 | 1197.95 | 1195.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 1271.95 | 1212.75 | 1202.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 1232.97 | 1241.52 | 1226.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1232.97 | 1241.52 | 1226.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1232.97 | 1241.52 | 1226.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1260.10 | 1247.02 | 1239.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 11:15:00 | 1270.88 | 1271.48 | 1271.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 11:15:00 | 1270.88 | 1271.48 | 1271.56 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 1274.38 | 1272.06 | 1271.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1275.15 | 1272.47 | 1272.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1270.38 | 1272.46 | 1272.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1270.38 | 1272.46 | 1272.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1270.38 | 1272.46 | 1272.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 1270.38 | 1272.46 | 1272.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 1268.78 | 1271.72 | 1271.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 14:15:00 | 1263.75 | 1269.48 | 1270.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 11:15:00 | 1270.35 | 1267.21 | 1268.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 11:15:00 | 1270.35 | 1267.21 | 1268.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1270.35 | 1267.21 | 1268.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 1270.35 | 1267.21 | 1268.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1273.00 | 1268.37 | 1269.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 13:00:00 | 1273.00 | 1268.37 | 1269.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 1269.03 | 1268.50 | 1269.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1264.60 | 1268.60 | 1269.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 1273.18 | 1263.04 | 1261.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 1273.18 | 1263.04 | 1261.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 1275.50 | 1268.10 | 1265.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 1263.90 | 1267.26 | 1265.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 1263.90 | 1267.26 | 1265.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1263.90 | 1267.26 | 1265.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 1263.90 | 1267.26 | 1265.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1264.22 | 1266.65 | 1265.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:30:00 | 1261.80 | 1266.65 | 1265.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1266.38 | 1266.60 | 1265.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1262.13 | 1266.60 | 1265.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1265.03 | 1266.28 | 1265.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1281.47 | 1266.28 | 1265.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1279.13 | 1268.85 | 1266.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 1285.00 | 1272.08 | 1268.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:30:00 | 1285.50 | 1277.91 | 1273.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 10:00:00 | 1284.78 | 1277.91 | 1273.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 09:15:00 | 1284.78 | 1284.63 | 1279.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1280.35 | 1283.78 | 1279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 1280.35 | 1283.78 | 1279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1278.43 | 1282.71 | 1279.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:30:00 | 1276.08 | 1282.71 | 1279.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1279.65 | 1282.09 | 1279.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-02 14:15:00 | 1271.38 | 1276.80 | 1277.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 1271.38 | 1276.80 | 1277.50 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 1280.63 | 1277.84 | 1277.79 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 1276.70 | 1277.61 | 1277.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 13:15:00 | 1273.70 | 1276.51 | 1277.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 15:15:00 | 1278.50 | 1276.79 | 1277.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 15:15:00 | 1278.50 | 1276.79 | 1277.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1278.50 | 1276.79 | 1277.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1278.28 | 1276.79 | 1277.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1277.85 | 1277.00 | 1277.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1268.90 | 1275.25 | 1276.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 12:15:00 | 1281.50 | 1277.60 | 1277.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 1281.50 | 1277.60 | 1277.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 1289.40 | 1280.75 | 1278.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 1305.83 | 1308.44 | 1301.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:45:00 | 1308.50 | 1308.44 | 1301.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 1309.25 | 1308.11 | 1302.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:30:00 | 1305.95 | 1308.11 | 1302.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1307.50 | 1309.22 | 1304.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1302.05 | 1309.22 | 1304.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1288.18 | 1305.01 | 1302.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 1288.18 | 1305.01 | 1302.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1288.88 | 1301.79 | 1301.64 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 1292.25 | 1299.88 | 1300.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 12:15:00 | 1282.93 | 1296.49 | 1299.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 15:15:00 | 1298.85 | 1296.30 | 1298.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 15:15:00 | 1298.85 | 1296.30 | 1298.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1298.85 | 1296.30 | 1298.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1301.25 | 1296.30 | 1298.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1297.83 | 1296.61 | 1298.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 1301.88 | 1296.61 | 1298.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1300.45 | 1297.38 | 1298.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:00:00 | 1300.45 | 1297.38 | 1298.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1299.08 | 1297.72 | 1298.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 1299.08 | 1297.72 | 1298.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 1303.33 | 1298.84 | 1298.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:00:00 | 1303.33 | 1298.84 | 1298.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 13:15:00 | 1304.75 | 1300.02 | 1299.51 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1295.05 | 1300.57 | 1300.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 10:15:00 | 1291.45 | 1298.75 | 1299.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 12:15:00 | 1304.55 | 1299.47 | 1299.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 1304.55 | 1299.47 | 1299.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1304.55 | 1299.47 | 1299.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 1304.55 | 1299.47 | 1299.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 13:15:00 | 1304.20 | 1300.41 | 1300.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 1310.47 | 1302.79 | 1301.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 1300.35 | 1302.31 | 1301.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 1300.35 | 1302.31 | 1301.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1300.35 | 1302.31 | 1301.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1300.35 | 1302.31 | 1301.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1303.15 | 1302.47 | 1301.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 1293.72 | 1302.47 | 1301.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1305.00 | 1302.98 | 1301.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:45:00 | 1296.83 | 1302.98 | 1301.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1307.43 | 1303.87 | 1302.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 1304.45 | 1303.87 | 1302.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1302.03 | 1307.64 | 1305.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 1305.00 | 1307.64 | 1305.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1300.00 | 1306.11 | 1304.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 1296.43 | 1306.11 | 1304.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 14:15:00 | 1298.75 | 1303.18 | 1303.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 11:15:00 | 1285.95 | 1298.55 | 1301.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1300.38 | 1295.14 | 1297.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 1300.38 | 1295.14 | 1297.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1300.38 | 1295.14 | 1297.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 1300.38 | 1295.14 | 1297.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1291.93 | 1294.50 | 1297.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:30:00 | 1301.28 | 1294.50 | 1297.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1292.08 | 1294.02 | 1296.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:00:00 | 1292.08 | 1294.02 | 1296.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1288.08 | 1292.83 | 1296.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1271.78 | 1291.73 | 1295.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 1241.88 | 1236.22 | 1235.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 11:15:00 | 1241.88 | 1236.22 | 1235.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 11:15:00 | 1247.50 | 1241.10 | 1238.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 14:15:00 | 1253.22 | 1256.09 | 1252.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-06 15:00:00 | 1253.22 | 1256.09 | 1252.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1255.00 | 1255.87 | 1252.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 1261.85 | 1255.87 | 1252.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1246.90 | 1255.46 | 1254.97 | SL hit (close<static) qty=1.00 sl=1250.50 alert=retest2 |

### Cycle 109 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 1244.00 | 1253.17 | 1253.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1237.47 | 1247.51 | 1249.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1245.33 | 1242.29 | 1245.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1245.33 | 1242.29 | 1245.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1245.33 | 1242.29 | 1245.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 1245.33 | 1242.29 | 1245.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1244.08 | 1242.65 | 1245.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 1245.80 | 1242.65 | 1245.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1241.40 | 1242.40 | 1244.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:30:00 | 1243.13 | 1242.40 | 1244.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1240.75 | 1242.01 | 1244.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:30:00 | 1241.95 | 1242.01 | 1244.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1241.83 | 1241.97 | 1243.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1241.83 | 1241.97 | 1243.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1233.63 | 1240.02 | 1242.73 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 1250.40 | 1242.33 | 1241.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 1253.65 | 1244.59 | 1242.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 1250.97 | 1255.50 | 1251.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 14:15:00 | 1250.97 | 1255.50 | 1251.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1250.97 | 1255.50 | 1251.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 1250.97 | 1255.50 | 1251.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 1252.50 | 1254.90 | 1251.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 1256.53 | 1254.90 | 1251.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:00:00 | 1257.80 | 1256.00 | 1253.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 13:15:00 | 1264.63 | 1267.88 | 1268.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 1264.63 | 1267.88 | 1268.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 14:15:00 | 1259.50 | 1266.21 | 1267.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 11:15:00 | 1254.50 | 1252.11 | 1256.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:00:00 | 1254.50 | 1252.11 | 1256.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1255.43 | 1252.77 | 1255.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 1256.55 | 1252.77 | 1255.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 1250.68 | 1252.35 | 1255.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:30:00 | 1257.75 | 1252.35 | 1255.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1252.08 | 1252.30 | 1255.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1252.08 | 1252.30 | 1255.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1250.88 | 1252.05 | 1254.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 1248.00 | 1252.05 | 1254.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 1248.60 | 1252.35 | 1253.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1256.09 | 1254.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 1264.50 | 1256.09 | 1254.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 1270.03 | 1258.88 | 1256.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1265.53 | 1265.84 | 1260.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1265.53 | 1265.84 | 1260.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1261.97 | 1265.00 | 1261.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 1261.97 | 1265.00 | 1261.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 1260.55 | 1264.11 | 1261.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:45:00 | 1261.05 | 1264.11 | 1261.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 1263.90 | 1264.07 | 1261.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:15:00 | 1266.55 | 1264.07 | 1261.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 09:15:00 | 1255.00 | 1264.19 | 1262.86 | SL hit (close<static) qty=1.00 sl=1259.68 alert=retest2 |

### Cycle 113 — SELL (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 11:15:00 | 1252.50 | 1260.09 | 1261.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 1250.00 | 1257.04 | 1259.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 13:15:00 | 1252.28 | 1251.83 | 1255.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 13:30:00 | 1252.15 | 1251.83 | 1255.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 15:15:00 | 1255.33 | 1252.73 | 1254.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 1246.50 | 1252.73 | 1254.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 12:15:00 | 1258.38 | 1253.10 | 1254.26 | SL hit (close>static) qty=1.00 sl=1257.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 1259.47 | 1255.23 | 1255.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1263.25 | 1257.36 | 1256.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 1264.97 | 1268.26 | 1264.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 12:15:00 | 1264.97 | 1268.26 | 1264.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1264.97 | 1268.26 | 1264.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:00:00 | 1264.97 | 1268.26 | 1264.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1265.47 | 1267.70 | 1264.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1265.47 | 1267.70 | 1264.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1263.53 | 1266.87 | 1264.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1263.53 | 1266.87 | 1264.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1264.97 | 1266.49 | 1264.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 1262.75 | 1266.49 | 1264.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1258.00 | 1264.79 | 1263.96 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 1256.85 | 1263.20 | 1263.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 11:15:00 | 1253.00 | 1261.16 | 1262.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 14:15:00 | 1262.35 | 1258.93 | 1260.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 1262.35 | 1258.93 | 1260.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 1262.35 | 1258.93 | 1260.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 15:00:00 | 1262.35 | 1258.93 | 1260.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 1262.00 | 1259.54 | 1260.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:15:00 | 1270.25 | 1259.54 | 1260.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 1275.78 | 1262.79 | 1262.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 09:15:00 | 1280.75 | 1272.55 | 1269.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 1273.95 | 1278.78 | 1274.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 1273.95 | 1278.78 | 1274.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1273.95 | 1278.78 | 1274.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 1270.95 | 1278.78 | 1274.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1272.97 | 1277.62 | 1274.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 1279.00 | 1277.62 | 1274.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 12:15:00 | 1338.50 | 1342.88 | 1342.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 1338.50 | 1342.88 | 1342.96 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 14:15:00 | 1350.05 | 1344.00 | 1343.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 1358.78 | 1347.03 | 1344.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 15:15:00 | 1370.00 | 1373.25 | 1365.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 09:15:00 | 1362.68 | 1373.25 | 1365.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1362.15 | 1371.03 | 1365.41 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 12:15:00 | 1347.60 | 1361.56 | 1362.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 1344.38 | 1355.98 | 1359.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 1351.20 | 1349.08 | 1353.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 13:30:00 | 1350.18 | 1349.08 | 1353.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1352.53 | 1349.77 | 1353.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 1352.53 | 1349.77 | 1353.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1353.60 | 1350.54 | 1353.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1335.43 | 1350.54 | 1353.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 09:15:00 | 1268.66 | 1279.06 | 1294.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-11 15:15:00 | 1257.50 | 1255.94 | 1262.16 | SL hit (close>ema200) qty=0.50 sl=1255.94 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1133.55 | 1130.42 | 1130.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1136.35 | 1131.61 | 1130.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1124.50 | 1132.86 | 1132.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1124.50 | 1132.86 | 1132.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1124.50 | 1132.86 | 1132.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1124.50 | 1132.86 | 1132.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1125.18 | 1131.32 | 1131.43 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 09:15:00 | 1139.00 | 1132.40 | 1131.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 13:15:00 | 1141.75 | 1137.20 | 1134.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 09:15:00 | 1140.00 | 1140.85 | 1137.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1140.00 | 1140.85 | 1137.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1140.00 | 1140.85 | 1137.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 1146.28 | 1140.85 | 1137.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:00:00 | 1145.72 | 1141.82 | 1137.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 13:15:00 | 1132.03 | 1139.05 | 1137.53 | SL hit (close<static) qty=1.00 sl=1135.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 1134.20 | 1136.99 | 1137.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 1129.53 | 1134.91 | 1136.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 15:15:00 | 1120.63 | 1120.44 | 1125.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 09:15:00 | 1115.95 | 1120.44 | 1125.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1106.83 | 1103.32 | 1108.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:00:00 | 1106.83 | 1103.32 | 1108.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1109.25 | 1104.93 | 1108.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 1115.03 | 1104.93 | 1108.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1117.88 | 1107.52 | 1109.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 1117.88 | 1107.52 | 1109.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1119.28 | 1109.87 | 1110.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 1119.28 | 1109.87 | 1110.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1118.25 | 1111.55 | 1110.94 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 1103.00 | 1109.70 | 1110.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1100.50 | 1107.86 | 1109.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 14:15:00 | 1105.70 | 1105.31 | 1107.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 14:15:00 | 1105.70 | 1105.31 | 1107.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 1105.70 | 1105.31 | 1107.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 1105.70 | 1105.31 | 1107.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 1104.00 | 1105.05 | 1107.05 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 1114.10 | 1108.43 | 1108.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 1118.45 | 1110.43 | 1109.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 1129.15 | 1130.34 | 1122.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 1129.15 | 1130.34 | 1122.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1131.40 | 1133.29 | 1129.22 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1122.58 | 1129.38 | 1129.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 14:15:00 | 1117.53 | 1125.89 | 1127.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1130.38 | 1125.57 | 1127.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1130.38 | 1125.57 | 1127.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1130.38 | 1125.57 | 1127.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1130.38 | 1125.57 | 1127.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1125.13 | 1125.48 | 1127.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 12:30:00 | 1122.03 | 1124.50 | 1126.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 12:00:00 | 1123.03 | 1120.63 | 1123.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 10:15:00 | 1129.03 | 1124.78 | 1124.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1129.03 | 1124.78 | 1124.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1131.25 | 1126.07 | 1124.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 1128.35 | 1129.23 | 1127.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 1128.35 | 1129.23 | 1127.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1123.45 | 1128.07 | 1126.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1123.45 | 1128.07 | 1126.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1129.30 | 1128.32 | 1127.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:15:00 | 1130.68 | 1128.32 | 1127.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 1116.33 | 1126.13 | 1126.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 1116.33 | 1126.13 | 1126.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 1113.33 | 1123.57 | 1125.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 11:15:00 | 1123.90 | 1123.64 | 1125.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 12:00:00 | 1123.90 | 1123.64 | 1125.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1122.75 | 1123.46 | 1125.06 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 1135.13 | 1126.66 | 1126.29 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 1111.53 | 1127.32 | 1127.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 1106.50 | 1114.12 | 1118.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 1124.35 | 1115.43 | 1118.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1124.35 | 1115.43 | 1118.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1124.35 | 1115.43 | 1118.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 1124.35 | 1115.43 | 1118.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1122.70 | 1116.88 | 1118.42 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 1122.50 | 1119.40 | 1119.39 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 1119.08 | 1119.34 | 1119.36 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 14:15:00 | 1120.00 | 1119.47 | 1119.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 1120.50 | 1119.67 | 1119.52 | Break + close above crossover candle high |

### Cycle 135 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1116.05 | 1118.95 | 1119.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 1113.47 | 1117.51 | 1118.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 1119.13 | 1114.10 | 1115.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 1119.13 | 1114.10 | 1115.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1119.13 | 1114.10 | 1115.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:45:00 | 1122.85 | 1114.10 | 1115.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1118.13 | 1114.91 | 1115.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 1118.13 | 1114.91 | 1115.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 1124.75 | 1116.87 | 1116.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 1126.78 | 1118.86 | 1117.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 1117.93 | 1120.05 | 1118.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1117.93 | 1120.05 | 1118.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1117.93 | 1120.05 | 1118.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 1119.22 | 1120.05 | 1118.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1116.50 | 1119.34 | 1118.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:30:00 | 1118.00 | 1119.34 | 1118.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1115.08 | 1118.49 | 1118.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 1115.08 | 1118.49 | 1118.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1116.53 | 1118.10 | 1117.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 1118.13 | 1117.85 | 1117.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 1108.78 | 1116.65 | 1117.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 1108.78 | 1116.65 | 1117.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 10:15:00 | 1106.50 | 1114.62 | 1116.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1084.58 | 1083.05 | 1090.16 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 13:00:00 | 1078.03 | 1082.03 | 1088.46 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1081.97 | 1082.06 | 1086.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 1078.65 | 1082.02 | 1085.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 14:45:00 | 1077.28 | 1080.11 | 1083.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 1085.65 | 1081.25 | 1083.48 | SL hit (close>ema400) qty=1.00 sl=1083.48 alert=retest1 |

### Cycle 138 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 1084.97 | 1081.96 | 1081.76 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 1076.05 | 1081.08 | 1081.49 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 1085.75 | 1081.95 | 1081.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 1088.25 | 1083.39 | 1082.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1101.50 | 1108.54 | 1101.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1101.50 | 1108.54 | 1101.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1101.50 | 1108.54 | 1101.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 1097.72 | 1108.54 | 1101.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1095.78 | 1105.99 | 1101.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1095.78 | 1105.99 | 1101.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1095.25 | 1103.84 | 1100.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1090.03 | 1103.84 | 1100.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 1089.58 | 1098.79 | 1098.86 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 1104.63 | 1099.43 | 1099.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 14:15:00 | 1106.53 | 1102.72 | 1100.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1101.97 | 1102.86 | 1101.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1101.97 | 1102.86 | 1101.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1101.97 | 1102.86 | 1101.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 1102.00 | 1102.86 | 1101.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1104.75 | 1103.24 | 1101.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 13:30:00 | 1107.15 | 1104.48 | 1102.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1106.20 | 1117.66 | 1119.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 1106.20 | 1117.66 | 1119.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1092.38 | 1108.20 | 1113.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 15:15:00 | 1100.50 | 1099.94 | 1105.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-16 09:15:00 | 1090.55 | 1099.94 | 1105.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1086.10 | 1097.17 | 1103.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:30:00 | 1081.38 | 1094.47 | 1102.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:00:00 | 1081.75 | 1089.57 | 1097.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 12:15:00 | 1110.00 | 1098.23 | 1098.56 | SL hit (close>static) qty=1.00 sl=1109.13 alert=retest2 |

### Cycle 144 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 1110.18 | 1100.62 | 1099.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 1112.53 | 1107.57 | 1105.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 1097.35 | 1107.24 | 1105.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 1097.35 | 1107.24 | 1105.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1097.35 | 1107.24 | 1105.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1097.35 | 1107.24 | 1105.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1100.00 | 1105.79 | 1105.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 1101.70 | 1105.79 | 1105.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 09:15:00 | 1098.03 | 1104.24 | 1104.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1098.03 | 1104.24 | 1104.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 09:15:00 | 1096.10 | 1102.40 | 1103.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 1102.00 | 1101.94 | 1103.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 11:15:00 | 1102.00 | 1101.94 | 1103.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1102.00 | 1101.94 | 1103.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:00:00 | 1102.00 | 1101.94 | 1103.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 1100.78 | 1101.71 | 1102.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 1099.75 | 1101.60 | 1102.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1099.55 | 1100.84 | 1102.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 1105.13 | 1101.70 | 1102.34 | SL hit (close>static) qty=1.00 sl=1104.80 alert=retest2 |

### Cycle 146 — BUY (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 12:15:00 | 1107.08 | 1103.41 | 1103.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 13:15:00 | 1110.13 | 1104.75 | 1103.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 1106.28 | 1106.54 | 1104.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 10:15:00 | 1106.28 | 1106.54 | 1104.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1106.28 | 1106.54 | 1104.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:45:00 | 1103.55 | 1106.54 | 1104.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 1104.08 | 1106.05 | 1104.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:00:00 | 1104.08 | 1106.05 | 1104.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 1096.18 | 1104.07 | 1104.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:00:00 | 1096.18 | 1104.07 | 1104.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 13:15:00 | 1090.70 | 1101.40 | 1102.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 14:15:00 | 1088.30 | 1098.78 | 1101.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 12:15:00 | 1080.38 | 1080.01 | 1085.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:30:00 | 1080.13 | 1080.01 | 1085.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1084.68 | 1080.95 | 1085.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 1084.68 | 1080.95 | 1085.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1091.35 | 1083.03 | 1086.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 1091.35 | 1083.03 | 1086.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1091.00 | 1084.62 | 1086.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 1092.15 | 1084.62 | 1086.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1096.25 | 1088.82 | 1088.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 1105.13 | 1093.76 | 1090.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 12:15:00 | 1161.65 | 1166.51 | 1152.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 13:00:00 | 1161.65 | 1166.51 | 1152.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1147.93 | 1159.69 | 1153.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:30:00 | 1150.00 | 1159.69 | 1153.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1147.60 | 1157.27 | 1152.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:30:00 | 1148.25 | 1157.27 | 1152.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1148.65 | 1153.50 | 1151.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 1146.22 | 1153.50 | 1151.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 1149.72 | 1152.09 | 1151.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 14:45:00 | 1147.00 | 1152.09 | 1151.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1149.97 | 1151.67 | 1151.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 1122.95 | 1151.67 | 1151.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 1115.90 | 1144.52 | 1147.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 10:15:00 | 1112.00 | 1118.28 | 1125.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 09:15:00 | 1116.20 | 1114.41 | 1119.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 09:15:00 | 1116.20 | 1114.41 | 1119.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1116.20 | 1114.41 | 1119.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1116.20 | 1114.41 | 1119.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1112.83 | 1114.09 | 1119.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 1117.60 | 1114.09 | 1119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1109.08 | 1110.46 | 1114.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:15:00 | 1106.35 | 1110.46 | 1114.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 1102.90 | 1109.69 | 1114.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1104.03 | 1095.62 | 1097.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 1105.68 | 1098.88 | 1098.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 12:15:00 | 1105.68 | 1098.88 | 1098.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 1109.20 | 1101.41 | 1099.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 10:15:00 | 1100.50 | 1102.51 | 1100.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 10:15:00 | 1100.50 | 1102.51 | 1100.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 1100.50 | 1102.51 | 1100.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 1100.50 | 1102.51 | 1100.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 1106.55 | 1103.32 | 1101.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 13:30:00 | 1109.58 | 1104.77 | 1102.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 14:15:00 | 1110.70 | 1104.77 | 1102.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 10:15:00 | 1110.00 | 1106.82 | 1103.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 1097.78 | 1103.28 | 1103.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 1097.78 | 1103.28 | 1103.54 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 12:15:00 | 1105.08 | 1103.10 | 1103.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 09:15:00 | 1109.18 | 1105.62 | 1104.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 1124.53 | 1131.05 | 1125.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 1124.53 | 1131.05 | 1125.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1124.53 | 1131.05 | 1125.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:30:00 | 1129.50 | 1131.05 | 1125.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 1123.97 | 1129.63 | 1124.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 1123.97 | 1129.63 | 1124.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 1123.83 | 1128.47 | 1124.89 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 13:15:00 | 1102.40 | 1120.39 | 1121.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 14:15:00 | 1094.50 | 1115.21 | 1119.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 1076.18 | 1075.43 | 1085.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 10:15:00 | 1089.50 | 1078.24 | 1086.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1089.50 | 1078.24 | 1086.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 1089.50 | 1078.24 | 1086.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1090.25 | 1080.64 | 1086.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 1090.43 | 1080.64 | 1086.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 1098.60 | 1089.79 | 1089.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 1107.08 | 1100.04 | 1095.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 14:15:00 | 1123.63 | 1125.00 | 1116.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 15:00:00 | 1123.63 | 1125.00 | 1116.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1136.00 | 1126.96 | 1118.48 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1100.22 | 1116.45 | 1118.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 11:15:00 | 1093.43 | 1102.30 | 1108.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1097.35 | 1090.30 | 1095.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 1097.35 | 1090.30 | 1095.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1097.35 | 1090.30 | 1095.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1097.35 | 1090.30 | 1095.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1100.95 | 1092.43 | 1096.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:30:00 | 1102.68 | 1092.43 | 1096.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 1097.58 | 1093.46 | 1096.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 10:45:00 | 1094.55 | 1096.45 | 1097.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 1105.53 | 1097.38 | 1096.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 1105.53 | 1097.38 | 1096.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 13:15:00 | 1110.40 | 1102.18 | 1099.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 1131.72 | 1132.34 | 1124.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 1131.72 | 1132.34 | 1124.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1130.10 | 1131.78 | 1125.48 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1120.75 | 1124.91 | 1125.10 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1131.58 | 1126.31 | 1125.71 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 15:15:00 | 1122.50 | 1125.29 | 1125.57 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1137.68 | 1127.77 | 1126.67 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 1117.40 | 1127.24 | 1127.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 1110.40 | 1123.87 | 1125.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 1106.45 | 1106.04 | 1112.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 10:00:00 | 1106.45 | 1106.04 | 1112.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1108.25 | 1106.48 | 1111.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 1110.80 | 1106.48 | 1111.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1116.70 | 1108.98 | 1111.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 1116.70 | 1108.98 | 1111.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 1119.05 | 1111.00 | 1112.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 1117.95 | 1111.00 | 1112.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 1120.00 | 1114.89 | 1114.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 09:15:00 | 1128.20 | 1117.55 | 1115.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 12:15:00 | 1120.65 | 1122.27 | 1118.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 12:15:00 | 1120.65 | 1122.27 | 1118.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 1120.65 | 1122.27 | 1118.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 1120.65 | 1122.27 | 1118.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1101.75 | 1120.57 | 1119.24 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 1100.80 | 1116.61 | 1117.56 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 1127.50 | 1119.09 | 1118.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 1137.05 | 1124.37 | 1120.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 10:15:00 | 1163.90 | 1164.89 | 1151.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 10:45:00 | 1164.15 | 1164.89 | 1151.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1189.15 | 1187.37 | 1180.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 1195.95 | 1189.09 | 1181.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:00:00 | 1208.30 | 1218.90 | 1211.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 1200.85 | 1208.75 | 1209.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1200.85 | 1208.75 | 1209.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 09:15:00 | 1187.00 | 1200.25 | 1203.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 1198.50 | 1195.67 | 1199.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1198.50 | 1195.67 | 1199.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1198.50 | 1195.67 | 1199.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:15:00 | 1201.05 | 1195.67 | 1199.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1200.10 | 1196.56 | 1199.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:15:00 | 1191.20 | 1197.38 | 1199.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:45:00 | 1194.65 | 1196.65 | 1198.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 10:15:00 | 1179.50 | 1167.46 | 1166.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1179.50 | 1167.46 | 1166.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1192.85 | 1178.42 | 1172.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1177.50 | 1181.41 | 1175.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1182.90 | 1182.09 | 1177.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:30:00 | 1180.20 | 1182.09 | 1177.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1184.15 | 1183.27 | 1179.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1196.35 | 1184.94 | 1182.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 1188.00 | 1196.60 | 1196.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1188.00 | 1196.60 | 1196.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1178.25 | 1192.93 | 1195.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1192.95 | 1190.87 | 1193.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1188.60 | 1190.41 | 1193.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1176.25 | 1192.26 | 1193.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1197.50 | 1185.12 | 1187.43 | SL hit (close>static) qty=1.00 sl=1195.05 alert=retest2 |

### Cycle 168 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1205.90 | 1189.28 | 1189.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1221.60 | 1202.96 | 1196.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1221.10 | 1224.24 | 1215.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:00:00 | 1221.10 | 1224.24 | 1215.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1214.45 | 1223.85 | 1218.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 1214.20 | 1223.85 | 1218.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1209.35 | 1220.95 | 1217.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 1209.35 | 1220.95 | 1217.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1209.25 | 1214.56 | 1215.07 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1218.90 | 1214.81 | 1214.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1221.70 | 1216.19 | 1215.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1216.65 | 1216.17 | 1215.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1210.05 | 1214.95 | 1214.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 1210.05 | 1214.95 | 1214.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1208.20 | 1213.60 | 1214.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 1207.40 | 1212.36 | 1213.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1201.10 | 1200.61 | 1203.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:45:00 | 1202.05 | 1200.61 | 1203.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1193.60 | 1199.21 | 1202.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 1202.60 | 1199.21 | 1202.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1194.00 | 1197.59 | 1201.44 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 1207.80 | 1200.83 | 1200.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 1212.60 | 1207.59 | 1204.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 1213.00 | 1213.41 | 1210.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 1213.00 | 1213.41 | 1210.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1210.60 | 1212.99 | 1210.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:45:00 | 1209.25 | 1212.99 | 1210.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1205.00 | 1211.39 | 1210.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1201.85 | 1209.63 | 1209.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1202.55 | 1208.22 | 1208.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1194.25 | 1205.42 | 1207.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1188.50 | 1188.05 | 1194.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1188.50 | 1188.05 | 1194.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1193.00 | 1189.27 | 1193.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 1193.00 | 1189.27 | 1193.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1194.50 | 1190.32 | 1193.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1194.50 | 1190.32 | 1193.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1193.70 | 1190.99 | 1193.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 1190.85 | 1191.63 | 1193.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1190.95 | 1192.49 | 1193.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1182.20 | 1174.59 | 1173.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1182.20 | 1174.59 | 1173.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 1191.55 | 1177.98 | 1175.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 13:15:00 | 1179.50 | 1181.66 | 1178.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 13:45:00 | 1179.65 | 1181.66 | 1178.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1179.85 | 1181.30 | 1178.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 1177.50 | 1181.30 | 1178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1179.00 | 1180.84 | 1178.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1184.70 | 1180.84 | 1178.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 1182.85 | 1181.59 | 1179.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 14:30:00 | 1181.50 | 1183.40 | 1181.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 1181.75 | 1183.40 | 1181.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1187.00 | 1184.12 | 1181.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1195.55 | 1184.12 | 1181.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1205.20 | 1217.89 | 1218.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 1205.20 | 1217.89 | 1218.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1195.60 | 1209.70 | 1214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1199.50 | 1198.24 | 1204.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:15:00 | 1200.60 | 1198.24 | 1204.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1196.30 | 1192.37 | 1196.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1196.30 | 1192.37 | 1196.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1196.00 | 1193.10 | 1196.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1200.00 | 1193.10 | 1196.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1200.80 | 1194.64 | 1196.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1200.70 | 1194.64 | 1196.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1201.55 | 1196.02 | 1197.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1200.50 | 1196.02 | 1197.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 1208.15 | 1199.98 | 1198.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 13:15:00 | 1211.30 | 1202.24 | 1200.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1207.00 | 1207.17 | 1204.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:30:00 | 1208.35 | 1207.17 | 1204.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1205.75 | 1206.89 | 1204.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 1204.50 | 1206.89 | 1204.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1209.70 | 1207.45 | 1204.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:30:00 | 1205.50 | 1207.45 | 1204.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1209.55 | 1212.41 | 1209.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1209.30 | 1212.41 | 1209.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1210.25 | 1211.98 | 1209.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1207.95 | 1211.98 | 1209.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1209.00 | 1211.38 | 1209.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 1208.40 | 1211.38 | 1209.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1207.45 | 1210.59 | 1209.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1207.45 | 1210.59 | 1209.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1204.75 | 1209.43 | 1209.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 1204.25 | 1209.43 | 1209.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1201.50 | 1207.84 | 1208.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1193.55 | 1203.28 | 1205.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1198.60 | 1198.31 | 1201.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 11:15:00 | 1198.70 | 1198.41 | 1200.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1198.70 | 1198.41 | 1200.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1198.70 | 1198.41 | 1200.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1206.10 | 1199.95 | 1201.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1206.10 | 1199.95 | 1201.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1204.65 | 1200.89 | 1201.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 1206.55 | 1200.89 | 1201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1208.15 | 1202.34 | 1202.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 1210.00 | 1203.87 | 1202.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 1224.40 | 1227.94 | 1221.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:45:00 | 1224.40 | 1227.94 | 1221.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1220.35 | 1225.94 | 1222.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1219.75 | 1225.94 | 1222.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1219.00 | 1224.55 | 1221.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1219.00 | 1224.55 | 1221.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1229.45 | 1229.89 | 1225.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1225.15 | 1229.89 | 1225.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1235.10 | 1231.35 | 1227.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1228.35 | 1231.35 | 1227.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1233.00 | 1234.03 | 1230.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1231.60 | 1234.03 | 1230.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1232.70 | 1233.76 | 1230.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 1233.50 | 1233.76 | 1230.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1225.55 | 1232.12 | 1230.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 1225.55 | 1232.12 | 1230.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1224.95 | 1230.69 | 1229.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1224.95 | 1230.69 | 1229.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 1221.15 | 1227.55 | 1228.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1212.50 | 1223.95 | 1225.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 1116.70 | 1116.44 | 1130.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 10:00:00 | 1116.70 | 1116.44 | 1130.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1125.25 | 1118.11 | 1122.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 1125.25 | 1118.11 | 1122.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1129.50 | 1120.39 | 1123.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 1129.50 | 1120.39 | 1123.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 1140.25 | 1126.39 | 1125.48 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1127.65 | 1132.30 | 1132.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 1124.20 | 1130.01 | 1131.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1116.85 | 1115.54 | 1120.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1118.00 | 1115.54 | 1120.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1120.50 | 1116.53 | 1120.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1111.50 | 1116.53 | 1120.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1109.00 | 1115.03 | 1119.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 1102.30 | 1110.78 | 1116.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 1105.00 | 1103.44 | 1109.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:45:00 | 1103.40 | 1103.49 | 1108.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1105.60 | 1104.75 | 1107.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1102.00 | 1104.20 | 1107.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 1099.00 | 1103.46 | 1106.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 1095.00 | 1094.28 | 1098.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1161.20 | 1105.49 | 1100.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1161.20 | 1105.49 | 1100.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1166.00 | 1117.59 | 1106.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1134.50 | 1139.70 | 1125.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 1131.50 | 1139.70 | 1125.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1163.20 | 1174.73 | 1168.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1163.20 | 1174.73 | 1168.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1170.20 | 1173.83 | 1168.93 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1159.00 | 1165.60 | 1166.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1152.30 | 1161.26 | 1164.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1175.00 | 1159.01 | 1161.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1170.20 | 1161.25 | 1162.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 1169.70 | 1161.25 | 1162.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 1176.50 | 1164.30 | 1163.51 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1158.60 | 1163.82 | 1163.93 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 13:15:00 | 1167.60 | 1164.50 | 1164.13 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 1160.70 | 1163.32 | 1163.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1150.50 | 1160.76 | 1162.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1168.80 | 1161.52 | 1161.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1171.90 | 1163.59 | 1162.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1194.50 | 1195.22 | 1186.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 1194.50 | 1195.22 | 1186.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1200.00 | 1207.88 | 1200.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 1197.90 | 1207.88 | 1200.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1197.10 | 1205.73 | 1200.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1197.10 | 1205.73 | 1200.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1193.60 | 1203.30 | 1199.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1193.60 | 1203.30 | 1199.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1202.00 | 1203.04 | 1200.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 1204.70 | 1203.04 | 1200.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 1205.80 | 1204.71 | 1202.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 1202.40 | 1204.25 | 1202.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:30:00 | 1202.80 | 1203.64 | 1201.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1192.20 | 1201.35 | 1201.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 1192.20 | 1201.35 | 1201.07 | SL hit (close<static) qty=1.00 sl=1193.30 alert=retest2 |

### Cycle 189 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1193.80 | 1199.84 | 1200.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1186.60 | 1197.19 | 1199.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 1195.80 | 1194.27 | 1196.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1198.90 | 1195.20 | 1196.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 1198.90 | 1195.20 | 1196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1201.60 | 1196.48 | 1197.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1201.60 | 1196.48 | 1197.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1200.60 | 1197.30 | 1197.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1208.20 | 1197.30 | 1197.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1199.04 | 1198.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1204.16 | 1200.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 1213.00 | 1217.81 | 1213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1215.40 | 1217.33 | 1213.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 1212.00 | 1217.33 | 1213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1213.30 | 1216.73 | 1214.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 1213.30 | 1216.73 | 1214.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1217.70 | 1216.92 | 1214.39 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1211.50 | 1213.62 | 1213.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 1207.10 | 1212.32 | 1213.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1200.50 | 1202.46 | 1205.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1201.90 | 1202.35 | 1205.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1198.20 | 1206.43 | 1206.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1198.90 | 1198.46 | 1201.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1165.00 | 1159.18 | 1159.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1165.00 | 1159.18 | 1159.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 1170.60 | 1166.19 | 1163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1177.70 | 1177.72 | 1172.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 1177.70 | 1177.72 | 1172.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1180.00 | 1180.33 | 1175.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1173.20 | 1180.33 | 1175.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1166.40 | 1177.54 | 1174.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1166.40 | 1177.54 | 1174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1169.40 | 1175.91 | 1174.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1172.70 | 1175.27 | 1174.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 1174.00 | 1175.52 | 1174.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1181.20 | 1186.64 | 1186.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1181.20 | 1186.64 | 1186.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1175.40 | 1184.39 | 1185.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1197.90 | 1182.53 | 1183.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1209.90 | 1188.01 | 1186.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 1222.90 | 1202.45 | 1193.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 1284.10 | 1284.19 | 1265.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:30:00 | 1282.10 | 1284.19 | 1265.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1276.60 | 1284.87 | 1278.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 1273.70 | 1284.87 | 1278.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1271.20 | 1282.14 | 1277.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1271.20 | 1282.14 | 1277.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1275.80 | 1280.87 | 1277.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1267.70 | 1280.87 | 1277.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1267.60 | 1275.33 | 1275.61 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 1280.80 | 1276.42 | 1276.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 1282.50 | 1278.36 | 1277.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 1278.60 | 1282.05 | 1279.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1282.90 | 1282.22 | 1280.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1281.50 | 1282.22 | 1280.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1280.00 | 1281.78 | 1280.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1284.20 | 1281.78 | 1280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1274.00 | 1280.22 | 1279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1274.00 | 1280.22 | 1279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1269.50 | 1278.08 | 1278.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1266.30 | 1275.72 | 1277.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA400 retest candle locked (from downside) |

### Cycle 198 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1279.90 | 1276.29 | 1275.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1281.60 | 1277.51 | 1276.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1275.90 | 1277.19 | 1276.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1277.50 | 1277.25 | 1276.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1278.50 | 1277.25 | 1276.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1275.20 | 1276.84 | 1276.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 1274.80 | 1276.84 | 1276.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1275.10 | 1276.49 | 1276.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:30:00 | 1272.50 | 1276.49 | 1276.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1271.10 | 1275.41 | 1275.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1259.20 | 1272.11 | 1274.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1266.80 | 1265.58 | 1269.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 1266.80 | 1265.58 | 1269.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1253.00 | 1263.58 | 1268.04 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 1275.00 | 1268.60 | 1267.85 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 1260.50 | 1266.52 | 1267.18 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1269.40 | 1266.47 | 1266.13 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 1261.70 | 1265.86 | 1265.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 12:15:00 | 1260.70 | 1264.46 | 1265.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1265.60 | 1264.67 | 1265.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1268.70 | 1265.47 | 1265.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 1267.50 | 1265.47 | 1265.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1266.20 | 1265.62 | 1265.57 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1263.30 | 1265.16 | 1265.36 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1280.60 | 1268.24 | 1266.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 1284.30 | 1277.62 | 1273.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1277.00 | 1279.75 | 1275.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1277.00 | 1279.75 | 1275.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1275.50 | 1278.90 | 1275.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1275.50 | 1278.90 | 1275.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1274.50 | 1278.02 | 1275.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1273.40 | 1278.02 | 1275.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1267.70 | 1275.95 | 1274.60 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 1268.20 | 1272.95 | 1273.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1255.90 | 1266.92 | 1269.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 1266.90 | 1265.48 | 1268.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 1266.90 | 1265.48 | 1268.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1269.50 | 1266.28 | 1268.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1269.50 | 1266.28 | 1268.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1265.00 | 1266.03 | 1267.90 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 1276.90 | 1268.79 | 1268.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 1278.50 | 1271.96 | 1270.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 1278.00 | 1278.27 | 1275.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 1280.10 | 1278.27 | 1275.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1279.10 | 1278.44 | 1275.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:30:00 | 1284.90 | 1280.47 | 1277.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1284.60 | 1281.05 | 1277.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 1271.40 | 1277.60 | 1277.33 | SL hit (close<static) qty=1.00 sl=1273.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 1275.70 | 1276.92 | 1277.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1268.50 | 1275.23 | 1276.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1268.30 | 1266.96 | 1270.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1262.60 | 1266.09 | 1269.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:30:00 | 1268.50 | 1266.09 | 1269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1277.10 | 1267.80 | 1270.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1277.10 | 1267.80 | 1270.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1278.30 | 1269.90 | 1270.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1279.00 | 1269.90 | 1270.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1280.60 | 1272.04 | 1271.70 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1266.30 | 1272.55 | 1272.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1263.20 | 1270.68 | 1272.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1262.30 | 1259.22 | 1262.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1262.30 | 1259.22 | 1262.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1259.00 | 1259.18 | 1262.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1257.90 | 1259.18 | 1262.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:00:00 | 1256.80 | 1258.70 | 1261.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:00:00 | 1258.40 | 1258.64 | 1261.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1258.20 | 1258.15 | 1260.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1258.10 | 1258.45 | 1260.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1238.90 | 1258.45 | 1260.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 1229.90 | 1220.46 | 1219.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1229.90 | 1220.46 | 1219.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 1233.00 | 1222.97 | 1220.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 1241.70 | 1242.41 | 1236.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 13:00:00 | 1241.70 | 1242.41 | 1236.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1241.50 | 1242.18 | 1237.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1234.80 | 1242.18 | 1237.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1237.00 | 1241.14 | 1237.78 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1228.20 | 1234.74 | 1235.55 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1242.30 | 1236.22 | 1235.76 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 15:15:00 | 1230.80 | 1235.07 | 1235.37 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 1239.70 | 1236.15 | 1235.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 1243.60 | 1237.64 | 1236.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 1253.60 | 1254.37 | 1250.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 1254.70 | 1254.37 | 1250.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.30 | 1255.36 | 1251.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 1262.40 | 1256.12 | 1251.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 1261.50 | 1257.04 | 1252.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1261.50 | 1258.47 | 1255.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 1280.40 | 1284.49 | 1284.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 1280.40 | 1284.49 | 1284.65 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1295.50 | 1286.69 | 1285.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 1308.40 | 1291.03 | 1287.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1301.40 | 1306.42 | 1299.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1330.80 | 1316.38 | 1307.52 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1306.60 | 1308.94 | 1309.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1305.70 | 1307.56 | 1308.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1307.30 | 1300.91 | 1302.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1314.40 | 1303.61 | 1303.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1314.40 | 1303.61 | 1303.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1314.10 | 1305.71 | 1304.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 1324.40 | 1315.35 | 1311.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 11:15:00 | 1314.60 | 1315.21 | 1311.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 1314.60 | 1315.21 | 1311.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1307.90 | 1313.75 | 1311.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1307.00 | 1313.75 | 1311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1305.90 | 1312.18 | 1310.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1305.90 | 1312.18 | 1310.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1307.90 | 1311.32 | 1310.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1316.90 | 1310.46 | 1310.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 1313.00 | 1310.85 | 1310.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 1311.30 | 1313.45 | 1312.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 1311.70 | 1314.09 | 1313.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1314.00 | 1314.07 | 1313.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 1311.80 | 1314.07 | 1313.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1309.20 | 1313.10 | 1313.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1309.20 | 1313.10 | 1313.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 1304.10 | 1311.30 | 1312.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1304.10 | 1311.30 | 1312.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 1295.20 | 1307.73 | 1310.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1290.50 | 1288.83 | 1296.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 1289.00 | 1288.83 | 1296.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1297.40 | 1290.54 | 1296.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1297.40 | 1290.54 | 1296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1296.40 | 1291.71 | 1296.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:30:00 | 1297.90 | 1291.71 | 1296.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1290.30 | 1291.43 | 1296.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:15:00 | 1293.50 | 1291.43 | 1296.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1301.50 | 1293.44 | 1296.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1301.50 | 1293.44 | 1296.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1305.80 | 1295.92 | 1297.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1308.20 | 1295.92 | 1297.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1309.00 | 1298.53 | 1298.45 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1295.30 | 1298.00 | 1298.32 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 1305.60 | 1298.50 | 1298.31 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1296.10 | 1298.02 | 1298.11 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 1302.50 | 1298.91 | 1298.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1305.60 | 1300.27 | 1299.24 | Break + close above crossover candle high |

### Cycle 227 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 1286.80 | 1298.49 | 1298.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 10:15:00 | 1277.20 | 1294.24 | 1296.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1292.00 | 1289.80 | 1293.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1295.40 | 1290.92 | 1293.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1283.50 | 1290.92 | 1293.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1288.50 | 1281.22 | 1286.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1302.60 | 1286.28 | 1287.52 | SL hit (close>static) qty=1.00 sl=1295.40 alert=retest2 |

### Cycle 228 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1303.70 | 1289.76 | 1288.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1333.50 | 1303.01 | 1295.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1311.00 | 1313.15 | 1302.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1313.20 | 1313.15 | 1302.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1305.00 | 1311.90 | 1304.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1305.00 | 1311.90 | 1304.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1301.00 | 1309.72 | 1303.85 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1271.00 | 1295.62 | 1298.38 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 1306.30 | 1298.49 | 1298.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1326.60 | 1309.11 | 1304.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1311.30 | 1311.53 | 1306.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:45:00 | 1310.70 | 1311.53 | 1306.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1311.10 | 1311.44 | 1307.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 1308.60 | 1311.44 | 1307.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1303.70 | 1309.89 | 1306.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1303.70 | 1309.89 | 1306.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1295.60 | 1307.04 | 1305.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1309.60 | 1307.04 | 1305.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1306.20 | 1306.87 | 1305.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1310.40 | 1307.57 | 1306.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1299.70 | 1304.51 | 1305.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 231 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1299.70 | 1304.51 | 1305.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1286.30 | 1300.08 | 1302.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1296.50 | 1295.32 | 1299.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 1296.50 | 1295.32 | 1299.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1305.00 | 1297.25 | 1299.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1305.00 | 1297.25 | 1299.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1300.00 | 1297.80 | 1299.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1295.00 | 1297.80 | 1299.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1297.60 | 1296.05 | 1298.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 1298.00 | 1296.05 | 1298.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1299.90 | 1296.82 | 1298.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 1299.90 | 1296.82 | 1298.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1295.80 | 1296.62 | 1298.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 1292.60 | 1296.21 | 1297.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1304.50 | 1297.68 | 1298.29 | SL hit (close>static) qty=1.00 sl=1301.50 alert=retest2 |

### Cycle 232 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 1301.90 | 1299.17 | 1298.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 1303.60 | 1300.05 | 1299.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1302.50 | 1305.78 | 1302.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1297.30 | 1304.08 | 1302.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 1297.30 | 1304.08 | 1302.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1301.00 | 1303.46 | 1302.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 1302.80 | 1303.46 | 1302.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 1305.20 | 1304.20 | 1302.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1296.10 | 1302.58 | 1302.34 | SL hit (close<static) qty=1.00 sl=1297.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1299.30 | 1302.61 | 1302.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 11:15:00 | 1291.20 | 1300.33 | 1301.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 1292.20 | 1289.76 | 1294.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:00:00 | 1292.20 | 1289.76 | 1294.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1291.40 | 1290.09 | 1294.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:30:00 | 1290.20 | 1289.87 | 1293.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:45:00 | 1287.60 | 1289.65 | 1293.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1284.60 | 1289.99 | 1292.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 1301.00 | 1289.10 | 1288.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 234 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1301.00 | 1289.10 | 1288.76 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1275.80 | 1286.99 | 1288.36 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1297.80 | 1288.91 | 1288.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1312.20 | 1296.75 | 1292.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1318.70 | 1321.20 | 1314.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:45:00 | 1315.50 | 1321.20 | 1314.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1320.00 | 1321.49 | 1316.55 | EMA400 retest candle locked (from upside) |

### Cycle 237 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1292.70 | 1313.50 | 1314.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1287.70 | 1305.22 | 1310.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1282.80 | 1280.81 | 1291.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1282.80 | 1280.81 | 1291.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1257.00 | 1249.28 | 1255.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 1257.40 | 1249.28 | 1255.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1258.90 | 1251.20 | 1255.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1258.90 | 1251.20 | 1255.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1261.10 | 1253.18 | 1256.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 1261.30 | 1253.18 | 1256.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1257.90 | 1253.71 | 1255.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1232.20 | 1253.71 | 1255.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1170.59 | 1186.44 | 1191.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1182.20 | 1175.66 | 1181.50 | SL hit (close>ema200) qty=0.50 sl=1175.66 alert=retest2 |

### Cycle 238 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1198.80 | 1185.82 | 1184.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1204.90 | 1189.64 | 1186.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1205.00 | 1205.73 | 1199.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:45:00 | 1204.70 | 1205.73 | 1199.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1190.50 | 1202.68 | 1198.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1190.50 | 1202.68 | 1198.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1194.10 | 1200.97 | 1198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1183.50 | 1200.97 | 1198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1185.50 | 1197.87 | 1197.13 | EMA400 retest candle locked (from upside) |

### Cycle 239 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1178.80 | 1194.06 | 1195.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1175.80 | 1185.75 | 1190.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 14:15:00 | 1181.20 | 1179.60 | 1184.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 15:00:00 | 1181.20 | 1179.60 | 1184.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1189.70 | 1176.89 | 1180.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 1189.70 | 1176.89 | 1180.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1194.00 | 1180.31 | 1182.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:30:00 | 1192.70 | 1180.31 | 1182.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 240 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1190.00 | 1183.98 | 1183.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1205.00 | 1188.63 | 1185.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 11:15:00 | 1210.30 | 1216.28 | 1207.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:00:00 | 1210.30 | 1216.28 | 1207.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1215.90 | 1215.10 | 1210.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 1229.50 | 1219.94 | 1214.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1226.40 | 1235.13 | 1227.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1228.20 | 1231.57 | 1227.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1225.70 | 1230.45 | 1227.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1225.40 | 1229.44 | 1227.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:15:00 | 1224.80 | 1229.44 | 1227.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1224.80 | 1228.51 | 1227.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1235.40 | 1228.51 | 1227.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 14:15:00 | 1352.45 | 1314.21 | 1295.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 11:15:00 | 1091.25 | 2023-05-18 10:15:00 | 1094.11 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2023-05-16 11:00:00 | 1092.49 | 2023-05-26 11:15:00 | 1076.38 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2023-05-16 11:30:00 | 1090.20 | 2023-05-26 11:15:00 | 1076.38 | STOP_HIT | 1.00 | 1.27% |
| SELL | retest2 | 2023-05-17 10:15:00 | 1092.39 | 2023-05-26 11:15:00 | 1076.38 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2023-05-17 11:15:00 | 1088.20 | 2023-05-26 11:15:00 | 1076.38 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2023-05-18 12:45:00 | 1086.46 | 2023-05-26 11:15:00 | 1076.38 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2023-05-31 10:15:00 | 1089.20 | 2023-05-31 11:15:00 | 1076.26 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2023-05-31 15:15:00 | 1088.95 | 2023-06-06 10:15:00 | 1089.76 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2023-06-01 09:30:00 | 1092.65 | 2023-06-06 11:15:00 | 1084.11 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-06-01 14:00:00 | 1092.50 | 2023-06-06 11:15:00 | 1084.11 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-06-06 09:45:00 | 1093.89 | 2023-06-06 11:15:00 | 1084.11 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2023-06-09 09:15:00 | 1109.40 | 2023-06-09 10:15:00 | 1103.96 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-12 09:15:00 | 1111.93 | 2023-06-20 10:15:00 | 1139.80 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2023-06-26 15:15:00 | 1125.25 | 2023-06-27 09:15:00 | 1135.99 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-06-30 14:15:00 | 1141.18 | 2023-07-03 11:15:00 | 1129.62 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-07-07 14:30:00 | 1150.93 | 2023-07-07 15:15:00 | 1143.45 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-07-11 13:15:00 | 1136.22 | 2023-07-11 14:15:00 | 1141.37 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-07-26 11:30:00 | 1136.26 | 2023-08-01 15:15:00 | 1132.50 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2023-07-27 09:30:00 | 1137.86 | 2023-08-01 15:15:00 | 1132.50 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2023-07-27 10:45:00 | 1137.20 | 2023-08-01 15:15:00 | 1132.50 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2023-08-04 15:15:00 | 1120.75 | 2023-08-21 10:15:00 | 1099.25 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2023-08-07 13:45:00 | 1122.49 | 2023-08-21 10:15:00 | 1099.25 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2023-08-07 15:00:00 | 1122.37 | 2023-08-21 10:15:00 | 1099.25 | STOP_HIT | 1.00 | 2.06% |
| SELL | retest2 | 2023-08-29 14:45:00 | 1101.00 | 2023-08-30 09:15:00 | 1105.45 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-09-04 10:15:00 | 1086.37 | 2023-09-05 11:15:00 | 1099.26 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2023-09-04 15:15:00 | 1085.01 | 2023-09-05 11:15:00 | 1099.26 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-09-14 09:15:00 | 1113.89 | 2023-09-21 10:15:00 | 1124.60 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2023-09-14 11:00:00 | 1113.12 | 2023-09-21 10:15:00 | 1124.60 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2023-09-27 14:30:00 | 1143.76 | 2023-09-28 10:15:00 | 1134.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-09-29 15:00:00 | 1124.24 | 2023-10-04 10:15:00 | 1135.63 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-10-03 09:30:00 | 1123.68 | 2023-10-04 10:15:00 | 1135.63 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-10-10 15:15:00 | 1150.01 | 2023-10-16 14:15:00 | 1155.48 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2023-11-15 15:15:00 | 1206.00 | 2023-11-16 10:15:00 | 1212.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2023-11-16 09:45:00 | 1206.10 | 2023-11-16 10:15:00 | 1212.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-11-29 12:30:00 | 1203.50 | 2023-11-30 14:15:00 | 1213.51 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-11-30 11:00:00 | 1201.76 | 2023-11-30 14:15:00 | 1213.51 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-11-30 14:00:00 | 1200.25 | 2023-11-30 14:15:00 | 1213.51 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-12-04 14:15:00 | 1222.28 | 2023-12-13 10:15:00 | 1246.05 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2023-12-04 15:15:00 | 1222.50 | 2023-12-13 10:15:00 | 1246.05 | STOP_HIT | 1.00 | 1.93% |
| SELL | retest2 | 2024-01-15 11:30:00 | 1277.88 | 2024-01-24 15:15:00 | 1258.25 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2024-01-15 12:45:00 | 1277.83 | 2024-01-24 15:15:00 | 1258.25 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-02-05 14:30:00 | 1234.33 | 2024-02-07 13:15:00 | 1247.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-02-07 10:15:00 | 1226.30 | 2024-02-07 13:15:00 | 1247.65 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-02-07 11:15:00 | 1233.93 | 2024-02-07 13:15:00 | 1247.65 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-02-15 13:30:00 | 1228.50 | 2024-02-15 15:15:00 | 1226.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-02-15 15:00:00 | 1228.88 | 2024-02-15 15:15:00 | 1226.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-02-27 14:30:00 | 1295.25 | 2024-02-28 15:15:00 | 1280.13 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-02-28 09:30:00 | 1296.90 | 2024-02-28 15:15:00 | 1280.13 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-02-28 10:30:00 | 1296.00 | 2024-02-28 15:15:00 | 1280.13 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-03-01 09:45:00 | 1306.72 | 2024-03-04 15:15:00 | 1290.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-03-01 13:45:00 | 1303.33 | 2024-03-04 15:15:00 | 1290.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-03-12 15:15:00 | 1297.50 | 2024-03-13 13:15:00 | 1286.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-03-14 09:15:00 | 1299.40 | 2024-03-18 09:15:00 | 1287.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-03-14 10:00:00 | 1299.53 | 2024-03-18 09:15:00 | 1287.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-03-14 11:30:00 | 1297.38 | 2024-03-18 09:15:00 | 1287.55 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-03-15 09:15:00 | 1313.00 | 2024-03-18 09:15:00 | 1287.55 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-03-28 09:15:00 | 1293.93 | 2024-04-03 09:15:00 | 1278.25 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-04-05 11:30:00 | 1277.25 | 2024-04-12 09:15:00 | 1276.78 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-05-10 09:15:00 | 1270.38 | 2024-05-14 09:15:00 | 1242.97 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-05-10 11:15:00 | 1269.95 | 2024-05-14 09:15:00 | 1242.97 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-05-10 13:00:00 | 1266.60 | 2024-05-14 09:15:00 | 1242.97 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-10 14:45:00 | 1266.47 | 2024-05-14 09:15:00 | 1242.97 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-05-17 09:15:00 | 1223.28 | 2024-05-18 09:15:00 | 1249.55 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1260.10 | 2024-06-18 11:15:00 | 1270.88 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-06-21 09:15:00 | 1264.60 | 2024-06-26 11:15:00 | 1273.18 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-06-28 11:00:00 | 1285.00 | 2024-07-02 14:15:00 | 1271.38 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-07-01 09:30:00 | 1285.50 | 2024-07-02 14:15:00 | 1271.38 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-01 10:00:00 | 1284.78 | 2024-07-02 14:15:00 | 1271.38 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-02 09:15:00 | 1284.78 | 2024-07-02 14:15:00 | 1271.38 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1268.90 | 2024-07-05 12:15:00 | 1281.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-24 09:15:00 | 1271.78 | 2024-08-01 11:15:00 | 1241.88 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2024-08-07 09:15:00 | 1261.85 | 2024-08-08 10:15:00 | 1246.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-20 09:15:00 | 1256.53 | 2024-08-26 13:15:00 | 1264.63 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2024-08-20 15:00:00 | 1257.80 | 2024-08-26 13:15:00 | 1264.63 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2024-08-30 10:15:00 | 1248.00 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-30 15:00:00 | 1248.60 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-04 12:15:00 | 1266.55 | 2024-09-05 09:15:00 | 1255.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-09 09:15:00 | 1246.50 | 2024-09-09 12:15:00 | 1258.38 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-09 12:30:00 | 1252.47 | 2024-09-09 13:15:00 | 1258.45 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-09-18 09:15:00 | 1279.00 | 2024-09-25 12:15:00 | 1338.50 | STOP_HIT | 1.00 | 4.65% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1335.43 | 2024-10-09 09:15:00 | 1268.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1335.43 | 2024-10-11 15:15:00 | 1257.50 | STOP_HIT | 0.50 | 5.84% |
| BUY | retest2 | 2024-11-11 10:15:00 | 1146.28 | 2024-11-11 13:15:00 | 1132.03 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-11-11 11:00:00 | 1145.72 | 2024-11-11 13:15:00 | 1132.03 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-29 12:30:00 | 1122.03 | 2024-12-03 10:15:00 | 1129.03 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-02 12:00:00 | 1123.03 | 2024-12-03 10:15:00 | 1129.03 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-12-04 13:15:00 | 1130.68 | 2024-12-05 09:15:00 | 1116.33 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-12-16 15:00:00 | 1118.13 | 2024-12-17 09:15:00 | 1108.78 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest1 | 2024-12-20 13:00:00 | 1078.03 | 2024-12-24 10:15:00 | 1085.65 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-23 13:00:00 | 1078.65 | 2024-12-30 10:15:00 | 1084.97 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-12-23 14:45:00 | 1077.28 | 2024-12-30 10:15:00 | 1084.97 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-26 13:45:00 | 1077.55 | 2024-12-30 10:15:00 | 1084.97 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-12-27 11:45:00 | 1078.33 | 2024-12-30 10:15:00 | 1084.97 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-01-08 13:30:00 | 1107.15 | 2025-01-14 09:15:00 | 1106.20 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-01-16 10:30:00 | 1081.38 | 2025-01-17 12:15:00 | 1110.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-01-16 14:00:00 | 1081.75 | 2025-01-17 12:15:00 | 1110.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-01-22 09:15:00 | 1101.70 | 2025-01-22 09:15:00 | 1098.03 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-01-23 15:00:00 | 1099.75 | 2025-01-24 10:15:00 | 1105.13 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1099.55 | 2025-01-24 10:15:00 | 1105.13 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-02-11 10:15:00 | 1106.35 | 2025-02-17 12:15:00 | 1105.68 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-02-11 11:15:00 | 1102.90 | 2025-02-17 12:15:00 | 1105.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1104.03 | 2025-02-17 12:15:00 | 1105.68 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-02-18 13:30:00 | 1109.58 | 2025-02-20 09:15:00 | 1097.78 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-02-18 14:15:00 | 1110.70 | 2025-02-20 09:15:00 | 1097.78 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-02-19 10:15:00 | 1110.00 | 2025-02-20 09:15:00 | 1097.78 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-03-19 10:45:00 | 1094.55 | 2025-03-20 10:15:00 | 1105.53 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-04-17 11:00:00 | 1195.95 | 2025-04-25 11:15:00 | 1200.85 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-04-24 12:00:00 | 1208.30 | 2025-04-25 11:15:00 | 1200.85 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-04-30 13:15:00 | 1191.20 | 2025-05-12 10:15:00 | 1179.50 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-04-30 14:45:00 | 1194.65 | 2025-05-12 10:15:00 | 1179.50 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1196.35 | 2025-05-20 13:15:00 | 1188.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1176.25 | 2025-05-23 09:15:00 | 1197.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-06-16 14:15:00 | 1190.85 | 2025-06-20 14:15:00 | 1182.20 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1190.95 | 2025-06-20 14:15:00 | 1182.20 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1184.70 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2025-06-24 09:45:00 | 1182.85 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-06-24 14:30:00 | 1181.50 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-06-24 15:00:00 | 1181.75 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1195.55 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-08-08 13:45:00 | 1102.30 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-08-11 13:15:00 | 1105.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-08-11 13:45:00 | 1103.40 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-08-12 10:15:00 | 1105.60 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2025-08-12 12:15:00 | 1099.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1095.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest2 | 2025-09-05 13:15:00 | 1204.70 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-08 09:30:00 | 1205.80 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-08 11:00:00 | 1202.40 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-08 11:30:00 | 1202.80 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-19 09:15:00 | 1198.20 | 2025-10-01 14:15:00 | 1165.00 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1198.90 | 2025-10-01 14:15:00 | 1165.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-10-08 12:00:00 | 1172.70 | 2025-10-14 10:15:00 | 1181.20 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-10-08 12:45:00 | 1174.00 | 2025-10-14 10:15:00 | 1181.20 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-11-21 12:30:00 | 1284.90 | 2025-11-24 11:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-21 13:30:00 | 1284.60 | 2025-11-24 11:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1257.90 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-12-02 10:00:00 | 1256.80 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2025-12-02 11:00:00 | 1258.40 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest2 | 2025-12-02 14:15:00 | 1258.20 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1238.90 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-12-24 11:15:00 | 1262.40 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-12-24 12:15:00 | 1261.50 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1261.50 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1316.90 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-16 09:45:00 | 1313.00 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-01-19 13:00:00 | 1311.30 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-01-20 10:00:00 | 1311.70 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1283.50 | 2026-01-30 09:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1288.50 | 2026-01-30 09:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1310.40 | 2026-02-05 13:15:00 | 1299.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-09 14:30:00 | 1292.60 | 2026-02-10 09:15:00 | 1304.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-11 14:15:00 | 1302.80 | 2026-02-12 10:15:00 | 1296.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-12 09:30:00 | 1305.20 | 2026-02-12 10:15:00 | 1296.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-02-12 11:45:00 | 1303.40 | 2026-02-13 10:15:00 | 1299.30 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1301.60 | 2026-02-13 10:15:00 | 1299.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-16 12:30:00 | 1290.20 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-16 13:45:00 | 1287.60 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1284.60 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1232.20 | 2026-03-23 10:15:00 | 1170.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1232.20 | 2026-03-24 11:15:00 | 1182.20 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2026-04-09 15:00:00 | 1229.50 | 2026-04-21 14:15:00 | 1352.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1226.40 | 2026-04-21 14:15:00 | 1349.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1228.20 | 2026-04-21 14:15:00 | 1351.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1225.70 | 2026-04-21 14:15:00 | 1348.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1235.40 | 2026-04-21 14:15:00 | 1358.94 | TARGET_HIT | 1.00 | 10.00% |
