# Tech Mahindra Ltd. (TECHM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1460.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 149 |
| ALERT2 | 145 |
| ALERT2_SKIP | 75 |
| ALERT3 | 424 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 184 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 192 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 197 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 64 / 133
- **Target hits / Stop hits / Partials:** 1 / 188 / 8
- **Avg / median % per leg:** 0.12% / -0.64%
- **Sum % (uncompounded):** 23.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 102 | 35 | 34.3% | 0 | 102 | 0 | 0.14% | 14.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.68% | -2.7% |
| BUY @ 3rd Alert (retest2) | 98 | 35 | 35.7% | 0 | 98 | 0 | 0.18% | 17.4% |
| SELL (all) | 95 | 29 | 30.5% | 1 | 86 | 8 | 0.09% | 8.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.84% | -0.8% |
| SELL @ 3rd Alert (retest2) | 94 | 29 | 30.9% | 1 | 85 | 8 | 0.10% | 9.8% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.71% | -3.6% |
| retest2 (combined) | 192 | 64 | 33.3% | 1 | 183 | 8 | 0.14% | 27.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 1044.55 | 1055.21 | 1056.48 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 1069.60 | 1057.32 | 1055.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 14:15:00 | 1072.00 | 1063.96 | 1059.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 13:15:00 | 1096.85 | 1099.42 | 1088.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 14:00:00 | 1096.85 | 1099.42 | 1088.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 1090.00 | 1096.14 | 1088.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 1093.40 | 1096.14 | 1088.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 1097.05 | 1096.32 | 1089.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 09:15:00 | 1111.30 | 1095.81 | 1094.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 12:00:00 | 1103.75 | 1110.11 | 1108.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 13:15:00 | 1103.10 | 1107.57 | 1107.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 13:15:00 | 1103.10 | 1107.57 | 1107.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 14:15:00 | 1099.55 | 1105.97 | 1106.93 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 09:15:00 | 1118.80 | 1107.42 | 1107.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-31 13:15:00 | 1122.40 | 1112.44 | 1109.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 13:15:00 | 1119.75 | 1120.76 | 1116.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 14:00:00 | 1119.75 | 1120.76 | 1116.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 1124.00 | 1123.14 | 1120.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:45:00 | 1121.00 | 1123.14 | 1120.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 1121.00 | 1122.71 | 1120.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:15:00 | 1114.50 | 1122.71 | 1120.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 1113.00 | 1120.77 | 1119.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 09:30:00 | 1113.50 | 1120.77 | 1119.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 1112.85 | 1119.19 | 1118.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 10:45:00 | 1108.15 | 1119.19 | 1118.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 11:15:00 | 1111.80 | 1117.71 | 1118.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 13:15:00 | 1110.25 | 1115.16 | 1117.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-07 12:15:00 | 1090.10 | 1089.22 | 1096.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-07 12:45:00 | 1091.15 | 1089.22 | 1096.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 15:15:00 | 1095.50 | 1091.63 | 1095.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-08 09:15:00 | 1091.85 | 1091.63 | 1095.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 1091.80 | 1091.66 | 1095.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 12:00:00 | 1085.85 | 1090.75 | 1094.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 10:15:00 | 1084.55 | 1078.05 | 1077.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 1084.55 | 1078.05 | 1077.77 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 09:15:00 | 1075.75 | 1078.35 | 1078.68 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 14:15:00 | 1082.05 | 1078.75 | 1078.63 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 10:15:00 | 1073.50 | 1077.78 | 1078.22 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-19 11:15:00 | 1091.15 | 1080.52 | 1079.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 14:15:00 | 1093.10 | 1085.29 | 1081.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 1098.55 | 1113.82 | 1110.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 1098.55 | 1113.82 | 1110.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 1098.55 | 1113.82 | 1110.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 10:00:00 | 1098.55 | 1113.82 | 1110.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 1111.00 | 1113.25 | 1110.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 12:00:00 | 1115.15 | 1113.63 | 1110.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 14:15:00 | 1113.75 | 1113.55 | 1111.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 10:00:00 | 1115.00 | 1114.43 | 1112.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-26 11:00:00 | 1115.25 | 1114.60 | 1112.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 1110.30 | 1116.25 | 1114.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 1110.30 | 1116.25 | 1114.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 1112.20 | 1115.44 | 1114.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 1111.75 | 1114.16 | 1113.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 1111.45 | 1113.62 | 1113.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:30:00 | 1110.20 | 1113.62 | 1113.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1117.30 | 1116.71 | 1115.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 09:30:00 | 1114.10 | 1116.71 | 1115.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 10:15:00 | 1120.20 | 1117.41 | 1115.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 11:00:00 | 1120.20 | 1117.41 | 1115.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 13:15:00 | 1114.35 | 1118.63 | 1116.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 14:00:00 | 1114.35 | 1118.63 | 1116.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 1108.70 | 1116.64 | 1116.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 1108.70 | 1116.64 | 1116.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-28 15:15:00 | 1108.25 | 1114.97 | 1115.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 15:15:00 | 1108.25 | 1114.97 | 1115.35 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 09:15:00 | 1129.55 | 1117.88 | 1116.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 12:15:00 | 1132.00 | 1126.41 | 1123.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-06 11:15:00 | 1166.10 | 1166.78 | 1154.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-06 11:30:00 | 1166.45 | 1166.78 | 1154.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 1179.95 | 1174.01 | 1163.07 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 09:15:00 | 1148.40 | 1160.25 | 1160.55 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 13:15:00 | 1162.20 | 1156.58 | 1156.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 10:15:00 | 1165.00 | 1159.60 | 1157.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 14:15:00 | 1161.10 | 1161.22 | 1159.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 15:00:00 | 1161.10 | 1161.22 | 1159.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 1157.00 | 1160.37 | 1159.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 09:15:00 | 1169.30 | 1160.37 | 1159.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 1180.95 | 1164.49 | 1161.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 10:15:00 | 1188.50 | 1164.49 | 1161.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-13 11:45:00 | 1188.75 | 1171.99 | 1165.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:15:00 | 1188.85 | 1174.75 | 1168.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-21 09:15:00 | 1205.75 | 1239.51 | 1239.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 09:15:00 | 1205.75 | 1239.51 | 1239.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 1160.40 | 1187.39 | 1203.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 1164.70 | 1160.82 | 1176.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-26 10:00:00 | 1164.70 | 1160.82 | 1176.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 11:15:00 | 1113.20 | 1102.35 | 1113.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 11:45:00 | 1114.30 | 1102.35 | 1113.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 12:15:00 | 1109.50 | 1103.78 | 1112.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 12:30:00 | 1113.40 | 1103.78 | 1112.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 1109.45 | 1104.91 | 1112.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:00:00 | 1109.45 | 1104.91 | 1112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 1115.85 | 1107.10 | 1112.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 15:00:00 | 1115.85 | 1107.10 | 1112.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 1119.00 | 1109.48 | 1113.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 1126.45 | 1109.48 | 1113.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 10:15:00 | 1137.00 | 1118.96 | 1117.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 12:15:00 | 1149.45 | 1128.09 | 1121.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 1134.00 | 1135.86 | 1128.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 10:15:00 | 1131.75 | 1135.86 | 1128.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 1127.10 | 1134.11 | 1128.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 1127.10 | 1134.11 | 1128.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 1133.85 | 1134.06 | 1128.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 12:30:00 | 1137.70 | 1133.93 | 1129.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 14:45:00 | 1137.25 | 1135.75 | 1130.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 14:45:00 | 1139.50 | 1135.71 | 1133.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 11:15:00 | 1216.10 | 1218.66 | 1218.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 1216.10 | 1218.66 | 1218.71 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 15:15:00 | 1220.80 | 1218.92 | 1218.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 09:15:00 | 1233.00 | 1221.73 | 1220.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-16 13:15:00 | 1224.75 | 1226.35 | 1223.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-16 14:00:00 | 1224.75 | 1226.35 | 1223.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 1227.50 | 1226.58 | 1223.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 14:30:00 | 1223.35 | 1226.58 | 1223.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 1221.95 | 1225.74 | 1223.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:45:00 | 1219.40 | 1225.74 | 1223.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 1220.10 | 1224.61 | 1223.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 11:15:00 | 1215.45 | 1224.61 | 1223.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 1224.15 | 1224.52 | 1223.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 13:30:00 | 1225.35 | 1225.10 | 1223.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 09:15:00 | 1213.90 | 1224.41 | 1224.05 | SL hit (close<static) qty=1.00 sl=1214.55 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 1209.60 | 1221.45 | 1222.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 09:15:00 | 1204.65 | 1212.30 | 1214.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 1207.10 | 1205.91 | 1209.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 1207.10 | 1205.91 | 1209.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 1207.10 | 1205.91 | 1209.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 10:45:00 | 1205.10 | 1205.59 | 1209.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:00:00 | 1204.05 | 1205.28 | 1208.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 13:15:00 | 1201.45 | 1205.41 | 1208.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 14:45:00 | 1204.10 | 1203.52 | 1207.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1182.80 | 1188.71 | 1195.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-08-29 12:15:00 | 1201.65 | 1195.40 | 1194.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 1201.65 | 1195.40 | 1194.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 1212.55 | 1200.87 | 1197.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 1197.65 | 1204.81 | 1201.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 1197.65 | 1204.81 | 1201.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 1197.65 | 1204.81 | 1201.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 1197.65 | 1204.81 | 1201.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 1197.00 | 1203.25 | 1201.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 09:15:00 | 1205.85 | 1203.25 | 1201.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 10:15:00 | 1254.55 | 1263.76 | 1264.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 10:15:00 | 1254.55 | 1263.76 | 1264.02 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 1281.50 | 1266.23 | 1264.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 12:15:00 | 1291.50 | 1281.97 | 1275.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 09:15:00 | 1287.00 | 1289.22 | 1281.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 1274.25 | 1287.67 | 1285.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1274.25 | 1287.67 | 1285.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:15:00 | 1274.15 | 1287.67 | 1285.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 1280.25 | 1286.19 | 1284.62 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 1277.60 | 1282.90 | 1283.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 1273.55 | 1281.03 | 1282.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 1291.80 | 1280.86 | 1281.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 10:15:00 | 1291.80 | 1280.86 | 1281.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 1291.80 | 1280.86 | 1281.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:00:00 | 1291.80 | 1280.86 | 1281.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 11:15:00 | 1294.00 | 1283.49 | 1282.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 09:15:00 | 1307.05 | 1292.27 | 1287.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-22 14:15:00 | 1305.00 | 1305.09 | 1296.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-22 15:00:00 | 1305.00 | 1305.09 | 1296.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 13:15:00 | 1297.05 | 1303.64 | 1299.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 14:00:00 | 1297.05 | 1303.64 | 1299.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 1299.50 | 1302.81 | 1299.84 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 1285.85 | 1296.20 | 1297.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 1281.20 | 1291.36 | 1294.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 11:15:00 | 1287.40 | 1285.00 | 1289.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 11:15:00 | 1287.40 | 1285.00 | 1289.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1287.40 | 1285.00 | 1289.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:45:00 | 1289.25 | 1285.00 | 1289.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 14:15:00 | 1288.00 | 1285.67 | 1288.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 15:00:00 | 1288.00 | 1285.67 | 1288.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 15:15:00 | 1289.90 | 1286.52 | 1288.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 09:15:00 | 1265.50 | 1286.52 | 1288.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 09:15:00 | 1202.22 | 1217.26 | 1229.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-04 15:15:00 | 1206.60 | 1205.21 | 1216.82 | SL hit (close>ema200) qty=0.50 sl=1205.21 alert=retest2 |

### Cycle 26 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 1215.90 | 1214.49 | 1214.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 09:15:00 | 1223.80 | 1216.45 | 1215.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 11:15:00 | 1215.05 | 1216.58 | 1215.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 11:15:00 | 1215.05 | 1216.58 | 1215.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 11:15:00 | 1215.05 | 1216.58 | 1215.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 12:00:00 | 1215.05 | 1216.58 | 1215.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 1214.70 | 1216.20 | 1215.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 12:45:00 | 1215.15 | 1216.20 | 1215.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 1214.15 | 1215.79 | 1215.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 14:00:00 | 1214.15 | 1215.79 | 1215.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 1211.50 | 1214.93 | 1215.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 1210.10 | 1213.97 | 1214.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1216.20 | 1214.41 | 1214.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1216.20 | 1214.41 | 1214.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1216.20 | 1214.41 | 1214.76 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 1221.35 | 1215.80 | 1215.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 1225.30 | 1218.55 | 1216.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 11:15:00 | 1225.30 | 1225.65 | 1221.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 12:00:00 | 1225.30 | 1225.65 | 1221.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 1228.00 | 1226.12 | 1222.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 14:30:00 | 1234.55 | 1227.63 | 1223.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 09:15:00 | 1220.50 | 1226.58 | 1223.78 | SL hit (close<static) qty=1.00 sl=1222.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 1201.75 | 1218.98 | 1221.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 15:15:00 | 1198.00 | 1211.85 | 1217.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 09:15:00 | 1195.55 | 1189.66 | 1195.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 1195.55 | 1189.66 | 1195.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 1195.55 | 1189.66 | 1195.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 1195.55 | 1189.66 | 1195.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 1192.65 | 1190.26 | 1195.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 1195.50 | 1190.26 | 1195.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 1195.10 | 1191.63 | 1195.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:45:00 | 1195.90 | 1191.63 | 1195.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 1196.00 | 1192.50 | 1195.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 13:30:00 | 1195.70 | 1192.50 | 1195.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 14:15:00 | 1197.50 | 1193.50 | 1195.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 14:30:00 | 1199.35 | 1193.50 | 1195.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 1188.45 | 1193.62 | 1195.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 11:45:00 | 1184.50 | 1191.50 | 1194.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 14:30:00 | 1186.45 | 1188.62 | 1191.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1125.27 | 1142.32 | 1154.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1127.13 | 1142.32 | 1154.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 1123.40 | 1121.86 | 1135.37 | SL hit (close>ema200) qty=0.50 sl=1121.86 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 14:15:00 | 1133.90 | 1127.10 | 1127.06 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 1122.95 | 1127.00 | 1127.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 09:15:00 | 1117.10 | 1125.02 | 1126.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 12:15:00 | 1124.85 | 1124.41 | 1125.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 12:15:00 | 1124.85 | 1124.41 | 1125.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 1124.85 | 1124.41 | 1125.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 13:00:00 | 1124.85 | 1124.41 | 1125.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 1120.40 | 1123.61 | 1125.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 14:30:00 | 1117.95 | 1121.88 | 1124.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 1132.40 | 1123.29 | 1124.57 | SL hit (close>static) qty=1.00 sl=1125.60 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 12:15:00 | 1134.00 | 1126.81 | 1125.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 1139.50 | 1132.18 | 1129.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 12:15:00 | 1146.95 | 1147.00 | 1142.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 12:45:00 | 1146.00 | 1147.00 | 1142.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 1141.90 | 1145.98 | 1142.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:00:00 | 1141.90 | 1145.98 | 1142.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 1139.10 | 1144.61 | 1142.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 15:00:00 | 1139.10 | 1144.61 | 1142.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 1138.85 | 1143.46 | 1142.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 1130.70 | 1143.46 | 1142.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 09:15:00 | 1124.90 | 1139.74 | 1140.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 10:15:00 | 1121.00 | 1136.00 | 1138.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 1130.85 | 1127.68 | 1132.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-10 09:45:00 | 1128.00 | 1127.68 | 1132.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 12:15:00 | 1134.30 | 1129.53 | 1132.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:00:00 | 1134.30 | 1129.53 | 1132.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 1136.25 | 1130.87 | 1132.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:45:00 | 1135.45 | 1130.87 | 1132.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 15:15:00 | 1139.10 | 1133.60 | 1133.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-12 18:15:00 | 1140.30 | 1134.94 | 1134.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-13 10:15:00 | 1134.15 | 1135.27 | 1134.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 10:15:00 | 1134.15 | 1135.27 | 1134.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 10:15:00 | 1134.15 | 1135.27 | 1134.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 10:45:00 | 1133.90 | 1135.27 | 1134.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 1131.75 | 1134.57 | 1134.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 12:00:00 | 1131.75 | 1134.57 | 1134.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 12:15:00 | 1128.50 | 1133.35 | 1133.68 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1168.45 | 1139.42 | 1136.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 1172.35 | 1146.01 | 1139.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 15:15:00 | 1200.85 | 1201.07 | 1188.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:15:00 | 1208.95 | 1201.07 | 1188.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 11:15:00 | 1206.35 | 1207.69 | 1200.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 1209.10 | 1207.97 | 1201.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 12:00:00 | 1209.10 | 1207.97 | 1201.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 1202.00 | 1206.51 | 1202.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-21 15:15:00 | 1202.00 | 1206.51 | 1202.78 | SL hit (close<ema400) qty=1.00 sl=1202.78 alert=retest1 |

### Cycle 37 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1204.30 | 1208.46 | 1208.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 14:15:00 | 1197.60 | 1205.66 | 1207.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 1201.90 | 1197.42 | 1201.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 1201.90 | 1197.42 | 1201.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1201.90 | 1197.42 | 1201.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 1201.90 | 1197.42 | 1201.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1208.00 | 1199.54 | 1201.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 1227.00 | 1199.54 | 1201.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 09:15:00 | 1229.45 | 1205.52 | 1204.33 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 1212.30 | 1218.42 | 1218.66 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 09:15:00 | 1227.90 | 1220.24 | 1219.20 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 1209.70 | 1227.03 | 1228.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 11:15:00 | 1199.00 | 1219.27 | 1224.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 1216.15 | 1214.70 | 1220.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 15:00:00 | 1216.15 | 1214.70 | 1220.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1247.60 | 1221.81 | 1223.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:45:00 | 1254.70 | 1221.81 | 1223.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 1247.35 | 1226.92 | 1225.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 1256.40 | 1232.81 | 1228.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 13:15:00 | 1300.90 | 1305.78 | 1287.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 14:00:00 | 1300.90 | 1305.78 | 1287.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 1290.00 | 1301.26 | 1288.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 1278.00 | 1295.65 | 1287.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1277.20 | 1291.96 | 1286.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 11:15:00 | 1271.05 | 1291.96 | 1286.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 1282.20 | 1284.71 | 1284.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 14:45:00 | 1279.70 | 1284.71 | 1284.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 1281.00 | 1283.97 | 1283.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 1294.80 | 1283.97 | 1283.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 1273.35 | 1282.50 | 1283.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 1273.35 | 1282.50 | 1283.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 1260.90 | 1278.18 | 1281.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1255.10 | 1253.74 | 1262.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 11:15:00 | 1254.95 | 1253.83 | 1260.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 1254.95 | 1253.83 | 1260.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:30:00 | 1257.05 | 1253.83 | 1260.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 1273.55 | 1256.49 | 1260.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 1273.55 | 1256.49 | 1260.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 1272.00 | 1259.59 | 1261.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 1263.55 | 1259.59 | 1261.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 1274.85 | 1262.64 | 1262.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 10:15:00 | 1283.95 | 1266.91 | 1264.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 11:15:00 | 1277.45 | 1278.77 | 1273.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 12:00:00 | 1277.45 | 1278.77 | 1273.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 1270.85 | 1277.18 | 1273.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 12:30:00 | 1272.15 | 1277.18 | 1273.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 1273.75 | 1276.50 | 1273.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:45:00 | 1272.50 | 1276.50 | 1273.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 14:15:00 | 1281.55 | 1277.51 | 1273.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 14:45:00 | 1275.45 | 1277.51 | 1273.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 09:15:00 | 1274.90 | 1277.11 | 1274.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:00:00 | 1274.90 | 1277.11 | 1274.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 1281.50 | 1277.99 | 1275.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 10:30:00 | 1274.65 | 1277.99 | 1275.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 1286.25 | 1281.26 | 1277.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:45:00 | 1288.55 | 1281.90 | 1278.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 14:15:00 | 1272.15 | 1277.04 | 1277.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 14:15:00 | 1272.15 | 1277.04 | 1277.34 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 12:15:00 | 1290.00 | 1279.74 | 1278.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 13:15:00 | 1300.60 | 1283.91 | 1280.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 1281.85 | 1286.41 | 1282.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 09:15:00 | 1281.85 | 1286.41 | 1282.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1281.85 | 1286.41 | 1282.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 1281.85 | 1286.41 | 1282.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1277.70 | 1284.67 | 1282.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 1279.10 | 1284.67 | 1282.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1275.75 | 1282.88 | 1281.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:45:00 | 1274.95 | 1282.88 | 1281.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 1279.00 | 1280.88 | 1280.86 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 14:15:00 | 1278.40 | 1280.38 | 1280.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 1250.20 | 1274.12 | 1277.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 14:15:00 | 1247.00 | 1245.01 | 1254.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-04 15:00:00 | 1247.00 | 1245.01 | 1254.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1263.10 | 1249.27 | 1255.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:30:00 | 1268.50 | 1249.27 | 1255.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 1262.85 | 1251.98 | 1255.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 11:00:00 | 1262.85 | 1251.98 | 1255.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1240.00 | 1250.12 | 1253.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 11:45:00 | 1231.90 | 1244.11 | 1250.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 09:15:00 | 1230.20 | 1238.85 | 1241.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 1253.40 | 1242.12 | 1241.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-01-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 09:15:00 | 1253.40 | 1242.12 | 1241.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-11 10:15:00 | 1255.60 | 1244.82 | 1243.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 09:15:00 | 1318.00 | 1326.70 | 1307.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 10:15:00 | 1316.95 | 1326.70 | 1307.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 1323.00 | 1322.53 | 1311.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 1313.50 | 1322.53 | 1311.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1327.60 | 1323.11 | 1314.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 14:00:00 | 1329.15 | 1323.60 | 1317.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 15:15:00 | 1333.00 | 1324.03 | 1317.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:30:00 | 1340.70 | 1333.97 | 1324.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 1327.75 | 1372.51 | 1376.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 10:15:00 | 1327.75 | 1372.51 | 1376.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 1319.90 | 1354.76 | 1367.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 1333.80 | 1324.24 | 1337.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-30 10:00:00 | 1333.80 | 1324.24 | 1337.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 1334.60 | 1327.23 | 1336.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 11:30:00 | 1338.00 | 1327.23 | 1336.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 12:15:00 | 1326.90 | 1327.17 | 1335.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-30 13:45:00 | 1322.30 | 1326.69 | 1334.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 10:15:00 | 1324.70 | 1324.35 | 1331.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 1320.35 | 1325.85 | 1328.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 11:00:00 | 1323.85 | 1326.30 | 1328.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1334.05 | 1321.85 | 1324.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:30:00 | 1332.95 | 1321.85 | 1324.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1342.70 | 1326.02 | 1326.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-02 10:15:00 | 1342.70 | 1326.02 | 1326.22 | SL hit (close>static) qty=1.00 sl=1337.85 alert=retest2 |

### Cycle 50 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 1347.05 | 1330.23 | 1328.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 15:15:00 | 1355.10 | 1347.92 | 1342.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 09:15:00 | 1345.65 | 1347.47 | 1342.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1345.65 | 1347.47 | 1342.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1345.65 | 1347.47 | 1342.86 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 14:15:00 | 1320.00 | 1337.87 | 1339.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 15:15:00 | 1315.00 | 1333.30 | 1337.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 10:15:00 | 1313.60 | 1313.35 | 1321.53 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 12:15:00 | 1307.20 | 1312.94 | 1320.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1318.15 | 1312.97 | 1317.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-12 09:15:00 | 1318.15 | 1312.97 | 1317.57 | SL hit (close>ema400) qty=1.00 sl=1317.57 alert=retest1 |

### Cycle 52 — BUY (started 2024-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 14:15:00 | 1327.95 | 1318.69 | 1318.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 15:15:00 | 1334.80 | 1321.91 | 1319.90 | Break + close above crossover candle high |

### Cycle 53 — SELL (started 2024-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 09:15:00 | 1302.95 | 1318.12 | 1318.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 10:15:00 | 1289.40 | 1312.38 | 1315.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 09:15:00 | 1305.80 | 1300.29 | 1306.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 1305.80 | 1300.29 | 1306.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 1305.80 | 1300.29 | 1306.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 09:45:00 | 1300.05 | 1300.29 | 1306.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 1299.55 | 1299.17 | 1304.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 13:30:00 | 1303.95 | 1299.17 | 1304.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1303.00 | 1300.12 | 1303.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 12:00:00 | 1299.80 | 1300.64 | 1303.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 13:45:00 | 1298.70 | 1301.06 | 1302.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 09:15:00 | 1292.90 | 1301.30 | 1302.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 11:15:00 | 1314.25 | 1302.99 | 1302.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 1314.25 | 1302.99 | 1302.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 14:15:00 | 1320.50 | 1312.07 | 1308.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 11:15:00 | 1315.35 | 1316.03 | 1311.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 12:15:00 | 1312.70 | 1315.37 | 1311.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 1312.70 | 1315.37 | 1311.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 13:00:00 | 1312.70 | 1315.37 | 1311.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 13:15:00 | 1303.40 | 1312.97 | 1311.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:00:00 | 1303.40 | 1312.97 | 1311.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 14:15:00 | 1299.75 | 1310.33 | 1310.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 14:45:00 | 1294.40 | 1310.33 | 1310.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1298.55 | 1307.97 | 1309.08 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 1324.60 | 1311.36 | 1310.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 1334.95 | 1318.26 | 1313.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 14:15:00 | 1321.00 | 1323.22 | 1318.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 15:00:00 | 1321.00 | 1323.22 | 1318.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1319.90 | 1322.55 | 1318.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 1302.60 | 1322.55 | 1318.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1307.20 | 1319.48 | 1317.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 1303.70 | 1319.48 | 1317.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 1300.85 | 1315.76 | 1315.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 1299.95 | 1307.94 | 1311.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 09:15:00 | 1301.45 | 1297.18 | 1302.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 1301.45 | 1297.18 | 1302.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1301.45 | 1297.18 | 1302.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:00:00 | 1301.45 | 1297.18 | 1302.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 1303.15 | 1298.37 | 1302.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:45:00 | 1300.60 | 1298.37 | 1302.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 1295.00 | 1297.70 | 1301.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 13:00:00 | 1290.60 | 1296.28 | 1300.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 14:30:00 | 1291.40 | 1294.91 | 1299.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-28 15:00:00 | 1288.80 | 1294.91 | 1299.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 11:15:00 | 1280.25 | 1272.43 | 1271.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 1280.25 | 1272.43 | 1271.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 14:15:00 | 1286.90 | 1277.16 | 1274.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 10:15:00 | 1279.20 | 1279.24 | 1276.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 11:00:00 | 1279.20 | 1279.24 | 1276.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1286.90 | 1288.23 | 1283.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 1284.80 | 1288.23 | 1283.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 1285.85 | 1287.75 | 1283.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:00:00 | 1285.85 | 1287.75 | 1283.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1285.35 | 1288.47 | 1285.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 1285.35 | 1288.47 | 1285.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 1285.50 | 1287.88 | 1285.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:30:00 | 1285.85 | 1287.88 | 1285.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 1278.60 | 1286.02 | 1284.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:30:00 | 1278.00 | 1286.02 | 1284.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 1278.80 | 1284.58 | 1284.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:30:00 | 1274.75 | 1284.58 | 1284.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 13:15:00 | 1277.90 | 1283.24 | 1283.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 15:15:00 | 1269.00 | 1279.44 | 1281.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1281.20 | 1278.37 | 1280.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 11:15:00 | 1281.20 | 1278.37 | 1280.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1281.20 | 1278.37 | 1280.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:30:00 | 1282.25 | 1278.37 | 1280.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 1277.95 | 1278.29 | 1280.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:30:00 | 1279.75 | 1278.29 | 1280.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 1294.95 | 1281.51 | 1281.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 1294.95 | 1281.51 | 1281.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 1290.30 | 1283.27 | 1282.32 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 11:15:00 | 1276.45 | 1281.22 | 1281.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-15 13:15:00 | 1272.20 | 1278.71 | 1280.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 14:15:00 | 1282.20 | 1279.41 | 1280.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 14:15:00 | 1282.20 | 1279.41 | 1280.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1282.20 | 1279.41 | 1280.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 1282.20 | 1279.41 | 1280.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1278.45 | 1279.22 | 1280.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:15:00 | 1283.40 | 1279.22 | 1280.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1276.05 | 1278.58 | 1279.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 10:30:00 | 1274.55 | 1277.38 | 1279.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 15:00:00 | 1272.90 | 1277.22 | 1278.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 1283.00 | 1268.72 | 1267.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 1283.00 | 1268.72 | 1267.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 1285.35 | 1276.21 | 1271.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 09:15:00 | 1269.65 | 1275.67 | 1272.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 09:15:00 | 1269.65 | 1275.67 | 1272.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 09:15:00 | 1269.65 | 1275.67 | 1272.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:30:00 | 1249.50 | 1275.67 | 1272.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 1266.80 | 1272.35 | 1271.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 14:00:00 | 1275.25 | 1272.14 | 1271.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 14:15:00 | 1259.10 | 1269.53 | 1270.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 1259.10 | 1269.53 | 1270.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 1254.00 | 1265.38 | 1268.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 1257.50 | 1254.44 | 1257.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 10:15:00 | 1257.50 | 1254.44 | 1257.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1257.50 | 1254.44 | 1257.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:45:00 | 1256.00 | 1254.44 | 1257.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 1256.25 | 1254.80 | 1257.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:30:00 | 1260.45 | 1254.80 | 1257.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 1260.00 | 1255.84 | 1257.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:30:00 | 1262.00 | 1255.84 | 1257.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 1258.70 | 1256.41 | 1257.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:15:00 | 1259.05 | 1256.41 | 1257.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1244.45 | 1254.02 | 1256.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 14:30:00 | 1262.10 | 1254.02 | 1256.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 1250.50 | 1252.19 | 1255.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 09:30:00 | 1255.45 | 1252.19 | 1255.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1242.50 | 1245.56 | 1249.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 10:45:00 | 1238.30 | 1244.56 | 1248.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-02 14:15:00 | 1238.50 | 1242.68 | 1246.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 09:15:00 | 1260.00 | 1244.68 | 1246.59 | SL hit (close>static) qty=1.00 sl=1251.95 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 11:15:00 | 1254.85 | 1248.84 | 1248.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 12:15:00 | 1263.05 | 1255.85 | 1252.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 10:15:00 | 1266.45 | 1267.05 | 1260.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-05 10:30:00 | 1267.00 | 1267.05 | 1260.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 1263.10 | 1265.75 | 1261.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:45:00 | 1262.90 | 1265.75 | 1261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 1262.80 | 1264.86 | 1261.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 14:30:00 | 1261.50 | 1264.86 | 1261.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 1264.95 | 1264.76 | 1262.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:45:00 | 1265.50 | 1264.76 | 1262.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 1263.05 | 1264.63 | 1262.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:45:00 | 1262.05 | 1264.63 | 1262.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 1265.45 | 1264.80 | 1263.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:15:00 | 1262.05 | 1264.80 | 1263.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 1262.05 | 1264.25 | 1262.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-09 09:15:00 | 1274.60 | 1264.25 | 1262.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 11:15:00 | 1256.80 | 1263.77 | 1263.17 | SL hit (close<static) qty=1.00 sl=1262.05 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 1246.70 | 1260.35 | 1261.68 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 11:15:00 | 1269.00 | 1261.93 | 1261.28 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 10:15:00 | 1254.30 | 1261.19 | 1261.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 1248.20 | 1258.30 | 1260.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 12:15:00 | 1190.00 | 1185.14 | 1196.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 13:00:00 | 1190.00 | 1185.14 | 1196.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 13:15:00 | 1201.35 | 1188.39 | 1197.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:00:00 | 1201.35 | 1188.39 | 1197.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1195.05 | 1189.72 | 1197.08 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 1208.30 | 1200.25 | 1200.05 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 10:15:00 | 1197.40 | 1201.47 | 1201.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 11:15:00 | 1195.90 | 1200.35 | 1201.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 13:15:00 | 1188.35 | 1186.95 | 1192.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 14:00:00 | 1188.35 | 1186.95 | 1192.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 1188.50 | 1187.26 | 1191.76 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 1312.30 | 1212.82 | 1202.64 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 10:15:00 | 1258.70 | 1265.39 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 11:15:00 | 1250.90 | 1262.49 | 1264.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 1256.85 | 1254.98 | 1259.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 1256.85 | 1254.98 | 1259.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1256.85 | 1254.98 | 1259.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:30:00 | 1259.00 | 1254.98 | 1259.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1262.60 | 1256.50 | 1259.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:00:00 | 1262.60 | 1256.50 | 1259.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 1257.55 | 1256.71 | 1259.16 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 10:15:00 | 1265.35 | 1260.74 | 1260.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 11:15:00 | 1268.70 | 1262.33 | 1261.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 1285.85 | 1287.10 | 1277.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 14:00:00 | 1285.85 | 1287.10 | 1277.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1268.55 | 1283.09 | 1278.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:00:00 | 1268.55 | 1283.09 | 1278.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1269.95 | 1280.47 | 1277.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 10:30:00 | 1263.35 | 1280.47 | 1277.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 1271.60 | 1278.47 | 1277.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:45:00 | 1270.30 | 1278.47 | 1277.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 1269.25 | 1276.63 | 1276.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:30:00 | 1267.00 | 1276.63 | 1276.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 1267.35 | 1274.77 | 1275.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 10:15:00 | 1265.10 | 1271.05 | 1273.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 12:15:00 | 1261.00 | 1260.57 | 1265.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-13 12:30:00 | 1260.15 | 1260.57 | 1265.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1263.90 | 1261.09 | 1264.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 1263.90 | 1261.09 | 1264.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1260.80 | 1261.03 | 1264.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 1261.00 | 1261.03 | 1264.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1257.40 | 1260.30 | 1263.52 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 1280.55 | 1265.07 | 1264.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1293.05 | 1277.79 | 1272.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 1306.05 | 1307.45 | 1296.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 14:45:00 | 1302.25 | 1307.45 | 1296.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1329.95 | 1335.00 | 1330.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:45:00 | 1328.55 | 1335.00 | 1330.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1321.15 | 1332.23 | 1329.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1321.15 | 1332.23 | 1329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1318.55 | 1329.49 | 1328.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:30:00 | 1325.30 | 1329.71 | 1328.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 1320.75 | 1329.80 | 1330.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 1320.75 | 1329.80 | 1330.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 1315.00 | 1323.69 | 1326.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 1246.65 | 1242.89 | 1259.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 1246.65 | 1242.89 | 1259.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1246.65 | 1242.89 | 1259.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1218.70 | 1245.13 | 1253.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 1234.05 | 1240.70 | 1250.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:45:00 | 1224.80 | 1236.61 | 1247.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:00:00 | 1235.20 | 1234.95 | 1244.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 1245.35 | 1237.03 | 1244.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 1245.35 | 1237.03 | 1244.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1247.95 | 1239.21 | 1245.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 1250.00 | 1239.21 | 1245.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1254.90 | 1242.35 | 1246.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:15:00 | 1262.05 | 1242.35 | 1246.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1266.35 | 1247.15 | 1247.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-05 10:15:00 | 1266.35 | 1247.15 | 1247.93 | SL hit (close>static) qty=1.00 sl=1261.80 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1280.80 | 1253.88 | 1250.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1293.80 | 1269.34 | 1260.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1339.55 | 1350.64 | 1324.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 1339.55 | 1350.64 | 1324.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1375.05 | 1382.42 | 1372.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:30:00 | 1370.25 | 1382.42 | 1372.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1376.65 | 1381.27 | 1372.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 1376.65 | 1381.27 | 1372.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1374.85 | 1379.98 | 1372.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:00:00 | 1374.85 | 1379.98 | 1372.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1375.85 | 1379.16 | 1373.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1383.15 | 1375.66 | 1372.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 13:00:00 | 1378.35 | 1377.94 | 1374.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1362.80 | 1373.34 | 1373.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 1362.80 | 1373.34 | 1373.63 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 11:15:00 | 1379.30 | 1373.85 | 1373.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 14:15:00 | 1381.20 | 1376.58 | 1375.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 1394.30 | 1395.68 | 1388.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 1394.30 | 1395.68 | 1388.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1399.55 | 1396.55 | 1389.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:30:00 | 1390.00 | 1396.55 | 1389.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1390.30 | 1394.57 | 1390.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:00:00 | 1396.40 | 1394.94 | 1390.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:45:00 | 1403.65 | 1397.11 | 1392.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 1396.00 | 1414.53 | 1411.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 1458.45 | 1467.61 | 1467.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 1458.45 | 1467.61 | 1467.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 1449.00 | 1460.14 | 1462.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 1463.35 | 1458.05 | 1460.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 1463.35 | 1458.05 | 1460.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1463.35 | 1458.05 | 1460.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 1463.35 | 1458.05 | 1460.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1462.45 | 1458.93 | 1460.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 1464.25 | 1458.93 | 1460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1454.95 | 1458.13 | 1460.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 1448.75 | 1456.26 | 1459.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 1480.10 | 1462.57 | 1460.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1480.10 | 1462.57 | 1460.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 1505.40 | 1471.14 | 1464.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 13:15:00 | 1495.25 | 1499.92 | 1489.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:00:00 | 1495.25 | 1499.92 | 1489.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1513.20 | 1502.46 | 1492.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:45:00 | 1519.60 | 1510.91 | 1502.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:15:00 | 1520.50 | 1510.91 | 1502.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:15:00 | 1519.65 | 1514.34 | 1505.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 1496.00 | 1510.01 | 1510.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 1496.00 | 1510.01 | 1510.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 1492.20 | 1506.45 | 1508.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 13:15:00 | 1495.20 | 1491.09 | 1496.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 1495.20 | 1491.09 | 1496.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1495.20 | 1491.09 | 1496.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1495.20 | 1491.09 | 1496.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1490.50 | 1490.97 | 1495.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1490.50 | 1490.97 | 1495.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1494.00 | 1491.58 | 1495.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 1507.90 | 1491.58 | 1495.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1520.00 | 1497.26 | 1497.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1520.00 | 1497.26 | 1497.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1521.55 | 1502.12 | 1500.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1529.80 | 1512.73 | 1505.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1514.30 | 1519.61 | 1511.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1514.30 | 1519.61 | 1511.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1514.30 | 1519.61 | 1511.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:15:00 | 1535.05 | 1521.45 | 1513.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:00:00 | 1535.25 | 1527.39 | 1517.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 1540.00 | 1527.95 | 1518.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 09:15:00 | 1500.10 | 1524.30 | 1518.94 | SL hit (close<static) qty=1.00 sl=1508.40 alert=retest2 |

### Cycle 83 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 1521.30 | 1522.92 | 1522.99 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 1526.70 | 1523.68 | 1523.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 09:15:00 | 1531.70 | 1525.81 | 1524.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 13:15:00 | 1547.00 | 1551.00 | 1545.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 13:15:00 | 1547.00 | 1551.00 | 1545.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1547.00 | 1551.00 | 1545.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 1545.75 | 1551.00 | 1545.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1544.45 | 1549.69 | 1545.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 1547.30 | 1549.69 | 1545.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1538.05 | 1547.36 | 1544.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1521.65 | 1547.36 | 1544.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1520.95 | 1542.08 | 1542.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1506.00 | 1519.96 | 1529.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1494.30 | 1473.14 | 1492.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1494.30 | 1473.14 | 1492.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1494.30 | 1473.14 | 1492.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1494.30 | 1473.14 | 1492.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1489.00 | 1476.31 | 1492.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:30:00 | 1484.80 | 1477.04 | 1491.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1482.55 | 1480.25 | 1490.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 1482.20 | 1480.74 | 1489.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 1482.70 | 1480.74 | 1489.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1501.55 | 1485.06 | 1490.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 1501.55 | 1485.06 | 1490.13 | SL hit (close>static) qty=1.00 sl=1499.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1500.40 | 1486.32 | 1485.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 12:15:00 | 1502.20 | 1489.49 | 1486.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1495.85 | 1496.84 | 1491.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1495.85 | 1496.84 | 1491.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1495.85 | 1496.84 | 1491.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1495.85 | 1496.84 | 1491.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1503.60 | 1510.48 | 1505.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 1503.60 | 1510.48 | 1505.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1501.30 | 1508.65 | 1504.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1501.30 | 1508.65 | 1504.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1500.00 | 1505.84 | 1504.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1518.85 | 1505.84 | 1504.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 12:15:00 | 1597.35 | 1602.81 | 1603.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 1597.35 | 1602.81 | 1603.53 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1619.70 | 1605.97 | 1604.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 1640.80 | 1622.58 | 1613.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 13:15:00 | 1627.35 | 1628.37 | 1621.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 14:00:00 | 1627.35 | 1628.37 | 1621.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1625.00 | 1627.26 | 1622.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1620.60 | 1627.26 | 1622.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1627.85 | 1627.38 | 1622.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 1621.95 | 1627.38 | 1622.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1627.15 | 1638.38 | 1630.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1627.15 | 1638.38 | 1630.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1634.00 | 1637.50 | 1631.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1630.50 | 1637.50 | 1631.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1640.80 | 1638.16 | 1632.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:00:00 | 1643.00 | 1639.43 | 1633.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 13:00:00 | 1644.80 | 1640.51 | 1634.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 1652.95 | 1640.84 | 1635.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 1650.20 | 1641.51 | 1639.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 1651.55 | 1643.52 | 1640.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 12:00:00 | 1658.85 | 1648.94 | 1644.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 1625.00 | 1644.15 | 1645.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 1625.00 | 1644.15 | 1645.47 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1645.00 | 1643.14 | 1642.90 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1634.95 | 1641.50 | 1642.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1628.45 | 1638.17 | 1640.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 1637.45 | 1636.69 | 1639.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 11:15:00 | 1637.45 | 1636.69 | 1639.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 1637.45 | 1636.69 | 1639.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:30:00 | 1638.80 | 1636.69 | 1639.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1629.15 | 1635.18 | 1638.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 13:30:00 | 1623.05 | 1633.13 | 1637.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 1612.00 | 1631.75 | 1635.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:15:00 | 1622.50 | 1628.75 | 1633.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 1623.15 | 1612.83 | 1612.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 12:15:00 | 1623.15 | 1612.83 | 1612.26 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1598.95 | 1609.95 | 1611.04 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 1632.00 | 1614.95 | 1612.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 1645.05 | 1620.97 | 1615.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 1648.60 | 1655.16 | 1645.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 1648.60 | 1655.16 | 1645.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1654.05 | 1653.55 | 1646.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1657.10 | 1653.55 | 1646.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:15:00 | 1657.20 | 1657.74 | 1651.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 1600.85 | 1644.98 | 1646.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 1600.85 | 1644.98 | 1646.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 1593.00 | 1634.58 | 1641.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1612.00 | 1602.70 | 1612.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 1612.00 | 1602.70 | 1612.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1612.00 | 1602.70 | 1612.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1612.00 | 1602.70 | 1612.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1616.15 | 1605.39 | 1612.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 1616.15 | 1605.39 | 1612.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 1624.00 | 1609.11 | 1613.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 1622.00 | 1609.11 | 1613.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1621.70 | 1612.71 | 1614.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 1618.80 | 1612.71 | 1614.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 1626.00 | 1615.37 | 1615.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 1627.20 | 1615.37 | 1615.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 1619.85 | 1616.26 | 1615.86 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 1608.20 | 1614.65 | 1615.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 11:15:00 | 1598.65 | 1611.45 | 1613.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 14:15:00 | 1608.15 | 1607.95 | 1611.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 15:00:00 | 1608.15 | 1607.95 | 1611.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1606.20 | 1607.60 | 1610.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 1602.90 | 1607.60 | 1610.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:30:00 | 1603.85 | 1606.93 | 1609.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 13:15:00 | 1631.55 | 1612.46 | 1611.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 1631.55 | 1612.46 | 1611.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 1636.85 | 1617.34 | 1613.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 1618.40 | 1620.98 | 1616.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 1618.40 | 1620.98 | 1616.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1618.40 | 1620.98 | 1616.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:30:00 | 1617.50 | 1620.98 | 1616.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1606.20 | 1618.02 | 1615.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 1612.15 | 1618.02 | 1615.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 1595.25 | 1613.47 | 1613.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 12:15:00 | 1592.40 | 1609.25 | 1611.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 1606.35 | 1605.10 | 1608.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1606.35 | 1605.10 | 1608.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1606.35 | 1605.10 | 1608.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:30:00 | 1597.25 | 1603.63 | 1606.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1626.95 | 1608.67 | 1608.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1626.95 | 1608.67 | 1608.31 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1587.20 | 1605.08 | 1607.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1570.00 | 1587.41 | 1597.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1629.15 | 1592.86 | 1596.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1629.15 | 1592.86 | 1596.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1629.15 | 1592.86 | 1596.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 1631.00 | 1592.86 | 1596.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1622.00 | 1598.69 | 1599.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 1628.10 | 1598.69 | 1599.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 11:15:00 | 1622.90 | 1603.53 | 1601.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 1632.85 | 1615.68 | 1611.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1619.75 | 1620.74 | 1615.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 1619.75 | 1620.74 | 1615.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1619.75 | 1620.74 | 1615.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:15:00 | 1619.45 | 1620.74 | 1615.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1618.00 | 1620.19 | 1615.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 1613.90 | 1620.19 | 1615.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1613.00 | 1618.75 | 1615.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1622.60 | 1618.75 | 1615.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 1609.45 | 1616.29 | 1614.91 | SL hit (close<static) qty=1.00 sl=1613.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 15:15:00 | 1612.00 | 1634.31 | 1635.55 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 1639.60 | 1636.90 | 1636.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 09:15:00 | 1663.25 | 1645.83 | 1641.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 10:15:00 | 1676.40 | 1678.27 | 1664.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:30:00 | 1680.00 | 1678.27 | 1664.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1669.95 | 1676.61 | 1665.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 1669.35 | 1676.61 | 1665.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1672.65 | 1674.99 | 1666.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 1672.65 | 1674.99 | 1666.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1675.00 | 1675.15 | 1668.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:45:00 | 1685.70 | 1668.33 | 1667.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 10:30:00 | 1677.90 | 1669.26 | 1668.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 1684.85 | 1675.33 | 1671.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1655.25 | 1681.52 | 1676.35 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1711.50 | 1722.57 | 1723.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1704.00 | 1718.85 | 1721.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 1699.30 | 1698.10 | 1705.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 15:00:00 | 1699.30 | 1698.10 | 1705.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1692.50 | 1697.88 | 1704.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 14:00:00 | 1683.85 | 1694.55 | 1700.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 15:00:00 | 1683.30 | 1692.30 | 1698.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:15:00 | 1599.66 | 1660.47 | 1681.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:15:00 | 1599.13 | 1660.47 | 1681.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1620.25 | 1618.53 | 1643.07 | SL hit (close>ema200) qty=0.50 sl=1618.53 alert=retest2 |

### Cycle 106 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1673.90 | 1639.04 | 1638.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 1683.60 | 1647.95 | 1642.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1666.70 | 1675.12 | 1661.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1666.70 | 1675.12 | 1661.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1651.10 | 1670.31 | 1660.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 1654.75 | 1670.31 | 1660.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1650.05 | 1666.26 | 1659.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 1665.95 | 1657.56 | 1656.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 10:15:00 | 1678.60 | 1687.59 | 1687.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1678.60 | 1687.59 | 1687.97 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 1692.00 | 1685.05 | 1684.30 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1653.10 | 1678.66 | 1681.47 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1707.60 | 1680.61 | 1678.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 1717.00 | 1687.89 | 1681.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1692.05 | 1697.32 | 1689.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1707.70 | 1697.32 | 1689.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:00:00 | 1700.80 | 1698.02 | 1690.28 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1692.45 | 1696.90 | 1690.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 1692.45 | 1696.90 | 1690.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 1689.10 | 1695.34 | 1690.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 1689.10 | 1695.34 | 1690.35 | SL hit (close<ema400) qty=1.00 sl=1690.35 alert=retest1 |

### Cycle 111 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 1716.05 | 1738.15 | 1740.83 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 1745.75 | 1730.21 | 1728.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 1746.60 | 1733.49 | 1730.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 1778.40 | 1781.18 | 1773.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 10:00:00 | 1778.40 | 1781.18 | 1773.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1777.00 | 1780.34 | 1774.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 1777.00 | 1780.34 | 1774.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1774.00 | 1779.07 | 1774.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 1774.00 | 1779.07 | 1774.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1774.90 | 1778.24 | 1774.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 1774.60 | 1778.24 | 1774.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 1774.25 | 1777.44 | 1774.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:00:00 | 1774.25 | 1777.44 | 1774.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1777.30 | 1777.41 | 1774.46 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 1763.65 | 1771.84 | 1772.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 1756.50 | 1768.77 | 1771.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 1764.55 | 1763.14 | 1766.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 1764.55 | 1763.14 | 1766.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1764.55 | 1763.14 | 1766.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:45:00 | 1766.90 | 1763.14 | 1766.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 1756.15 | 1761.74 | 1766.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 1754.90 | 1761.26 | 1765.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:15:00 | 1754.20 | 1760.13 | 1764.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:15:00 | 1755.15 | 1760.70 | 1764.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 1800.60 | 1767.79 | 1766.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 1800.60 | 1767.79 | 1766.90 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 1776.75 | 1779.92 | 1779.95 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 1787.00 | 1779.95 | 1779.82 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 1778.60 | 1779.68 | 1779.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 1768.95 | 1777.53 | 1778.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 1775.70 | 1772.71 | 1775.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1775.70 | 1772.71 | 1775.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1775.70 | 1772.71 | 1775.27 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 14:15:00 | 1779.15 | 1776.78 | 1776.64 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 1775.00 | 1776.42 | 1776.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1757.30 | 1772.60 | 1774.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 1716.95 | 1709.43 | 1729.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 1716.95 | 1709.43 | 1729.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1718.25 | 1711.83 | 1722.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:00:00 | 1704.80 | 1712.23 | 1719.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1705.70 | 1710.44 | 1717.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 1705.80 | 1705.31 | 1708.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1704.30 | 1708.63 | 1709.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1712.50 | 1709.40 | 1709.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 1714.95 | 1710.51 | 1710.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 1714.95 | 1710.51 | 1710.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 12:15:00 | 1723.65 | 1713.91 | 1711.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1683.80 | 1715.08 | 1713.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1683.80 | 1715.08 | 1713.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1683.80 | 1715.08 | 1713.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 1683.80 | 1715.08 | 1713.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 10:15:00 | 1695.45 | 1711.15 | 1712.23 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1716.90 | 1707.84 | 1707.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 1725.85 | 1713.50 | 1710.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 1695.20 | 1712.99 | 1711.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 1695.20 | 1712.99 | 1711.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1695.20 | 1712.99 | 1711.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 1695.20 | 1712.99 | 1711.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 1690.00 | 1708.39 | 1709.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 1671.40 | 1685.82 | 1693.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1657.85 | 1654.60 | 1668.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 1657.85 | 1654.60 | 1668.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1665.00 | 1658.42 | 1667.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1654.40 | 1658.42 | 1667.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 1651.80 | 1656.12 | 1665.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 1676.05 | 1654.79 | 1659.62 | SL hit (close>static) qty=1.00 sl=1669.95 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 11:15:00 | 1706.85 | 1671.64 | 1666.84 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 1656.25 | 1672.70 | 1674.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 1633.45 | 1664.40 | 1670.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 1660.85 | 1660.74 | 1666.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 13:45:00 | 1661.45 | 1660.74 | 1666.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1647.60 | 1658.11 | 1664.61 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-01-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 13:15:00 | 1674.20 | 1665.24 | 1665.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 14:15:00 | 1677.85 | 1667.76 | 1666.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1677.15 | 1681.40 | 1676.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 11:15:00 | 1677.15 | 1681.40 | 1676.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 1677.15 | 1681.40 | 1676.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 1678.55 | 1681.40 | 1676.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 1671.00 | 1679.32 | 1676.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 1671.00 | 1679.32 | 1676.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1659.15 | 1675.29 | 1674.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 1659.15 | 1675.29 | 1674.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 14:15:00 | 1658.30 | 1671.89 | 1673.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 09:15:00 | 1653.90 | 1666.01 | 1670.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 1670.05 | 1660.81 | 1665.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 13:15:00 | 1670.05 | 1660.81 | 1665.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1670.05 | 1660.81 | 1665.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 1670.05 | 1660.81 | 1665.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1676.70 | 1663.99 | 1666.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 1678.05 | 1663.99 | 1666.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1675.00 | 1666.19 | 1667.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1689.20 | 1666.19 | 1667.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1666.90 | 1666.89 | 1667.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:45:00 | 1668.00 | 1666.89 | 1667.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 1680.90 | 1669.70 | 1668.77 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 1638.45 | 1664.17 | 1666.64 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 12:15:00 | 1674.20 | 1666.39 | 1666.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 14:15:00 | 1685.50 | 1670.79 | 1668.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 09:15:00 | 1695.55 | 1715.46 | 1707.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 1695.55 | 1715.46 | 1707.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1695.55 | 1715.46 | 1707.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:15:00 | 1688.00 | 1715.46 | 1707.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1684.85 | 1709.34 | 1704.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 11:00:00 | 1684.85 | 1709.34 | 1704.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1667.45 | 1700.96 | 1701.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 1657.95 | 1692.36 | 1697.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1673.45 | 1659.65 | 1669.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1673.45 | 1659.65 | 1669.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1673.45 | 1659.65 | 1669.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1673.45 | 1659.65 | 1669.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1674.25 | 1662.57 | 1670.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1674.25 | 1662.57 | 1670.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1672.05 | 1664.47 | 1670.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 1672.95 | 1664.47 | 1670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1665.00 | 1664.57 | 1669.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:30:00 | 1675.30 | 1664.57 | 1669.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1684.55 | 1668.57 | 1671.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 1684.55 | 1668.57 | 1671.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1681.90 | 1671.23 | 1672.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:15:00 | 1684.70 | 1671.23 | 1672.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1684.70 | 1673.93 | 1673.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1699.30 | 1679.00 | 1675.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1664.10 | 1680.77 | 1678.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 1664.10 | 1680.77 | 1678.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1664.10 | 1680.77 | 1678.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1664.10 | 1680.77 | 1678.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1671.55 | 1678.93 | 1677.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:45:00 | 1665.70 | 1678.93 | 1677.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1680.80 | 1679.63 | 1678.17 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-01-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 12:15:00 | 1673.25 | 1677.45 | 1677.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 10:15:00 | 1663.90 | 1673.14 | 1675.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 1658.90 | 1650.46 | 1658.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 1658.90 | 1650.46 | 1658.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 1658.90 | 1650.46 | 1658.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 1658.90 | 1650.46 | 1658.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 1659.25 | 1652.22 | 1658.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 15:15:00 | 1651.05 | 1652.87 | 1658.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 1650.75 | 1652.97 | 1657.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:30:00 | 1652.25 | 1652.41 | 1656.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:45:00 | 1652.40 | 1653.15 | 1655.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 1657.80 | 1653.70 | 1655.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:45:00 | 1660.15 | 1653.70 | 1655.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 1660.30 | 1655.02 | 1655.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 13:45:00 | 1662.20 | 1655.02 | 1655.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-05 14:15:00 | 1661.35 | 1656.29 | 1656.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 14:15:00 | 1661.35 | 1656.29 | 1656.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 1669.80 | 1660.87 | 1658.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1661.10 | 1662.07 | 1659.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 1661.10 | 1662.07 | 1659.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1666.80 | 1665.96 | 1662.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 1679.60 | 1668.53 | 1664.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 15:15:00 | 1671.15 | 1678.35 | 1678.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 15:15:00 | 1671.15 | 1678.35 | 1678.85 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 09:15:00 | 1682.85 | 1679.25 | 1679.21 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 10:15:00 | 1677.65 | 1678.93 | 1679.07 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 11:15:00 | 1684.90 | 1680.12 | 1679.60 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 13:15:00 | 1677.15 | 1678.98 | 1679.14 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 1684.25 | 1680.04 | 1679.57 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1661.15 | 1676.77 | 1678.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 1645.20 | 1661.48 | 1669.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 1659.35 | 1655.63 | 1663.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 13:15:00 | 1659.35 | 1655.63 | 1663.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1659.35 | 1655.63 | 1663.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 1662.45 | 1655.63 | 1663.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1667.35 | 1657.97 | 1663.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1667.35 | 1657.97 | 1663.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1665.95 | 1659.57 | 1664.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1687.30 | 1659.57 | 1664.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1684.55 | 1664.57 | 1665.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:45:00 | 1684.10 | 1664.57 | 1665.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 1680.80 | 1667.81 | 1667.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 12:15:00 | 1688.80 | 1674.21 | 1670.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 14:15:00 | 1685.55 | 1693.70 | 1686.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 14:15:00 | 1685.55 | 1693.70 | 1686.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 1685.55 | 1693.70 | 1686.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:30:00 | 1682.15 | 1693.70 | 1686.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 1689.00 | 1692.76 | 1686.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 1680.90 | 1692.76 | 1686.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1685.85 | 1691.38 | 1686.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:45:00 | 1683.70 | 1691.38 | 1686.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1675.55 | 1688.21 | 1685.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 1673.10 | 1688.21 | 1685.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 1670.80 | 1684.73 | 1684.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:45:00 | 1673.10 | 1684.73 | 1684.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 1663.15 | 1680.42 | 1682.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 13:15:00 | 1659.75 | 1676.28 | 1680.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1589.30 | 1588.47 | 1604.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 1589.30 | 1588.47 | 1604.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1515.15 | 1495.45 | 1506.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 1515.15 | 1495.45 | 1506.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1512.75 | 1498.91 | 1507.43 | EMA400 retest candle locked (from downside) |

### Cycle 144 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1538.20 | 1512.48 | 1512.47 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 14:15:00 | 1504.15 | 1515.94 | 1517.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 09:15:00 | 1496.00 | 1509.71 | 1513.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 13:15:00 | 1494.70 | 1493.76 | 1499.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 1494.70 | 1493.76 | 1499.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1485.65 | 1476.62 | 1484.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 1459.60 | 1476.62 | 1484.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1448.00 | 1470.89 | 1481.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 1446.00 | 1470.89 | 1481.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 10:45:00 | 1442.80 | 1441.17 | 1443.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 10:15:00 | 1433.40 | 1419.54 | 1418.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 1433.40 | 1419.54 | 1418.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 11:15:00 | 1439.45 | 1423.52 | 1420.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 15:15:00 | 1451.20 | 1453.95 | 1444.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-26 09:15:00 | 1452.65 | 1453.95 | 1444.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1445.30 | 1452.22 | 1444.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 1442.40 | 1452.22 | 1444.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1434.15 | 1448.61 | 1443.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 11:00:00 | 1434.15 | 1448.61 | 1443.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 1424.45 | 1443.77 | 1441.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 1424.45 | 1443.77 | 1441.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 1420.95 | 1439.21 | 1439.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1417.40 | 1432.01 | 1436.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1424.35 | 1421.97 | 1427.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 1424.35 | 1421.97 | 1427.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1427.45 | 1423.07 | 1427.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 1417.35 | 1423.07 | 1427.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:00:00 | 1420.90 | 1420.58 | 1425.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 1419.40 | 1421.29 | 1425.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 1419.80 | 1419.82 | 1423.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1417.50 | 1403.46 | 1409.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1415.15 | 1403.46 | 1409.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1419.75 | 1406.72 | 1410.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1419.75 | 1406.72 | 1410.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1421.35 | 1411.16 | 1412.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1421.35 | 1411.16 | 1412.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-02 13:15:00 | 1421.65 | 1413.26 | 1413.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1421.65 | 1413.26 | 1413.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 1424.50 | 1417.03 | 1414.95 | Break + close above crossover candle high |

### Cycle 149 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 1387.00 | 1411.03 | 1412.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 10:15:00 | 1375.50 | 1403.92 | 1409.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1289.15 | 1286.19 | 1319.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1289.15 | 1286.19 | 1319.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1296.80 | 1289.40 | 1315.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1275.45 | 1311.06 | 1317.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:45:00 | 1292.05 | 1289.56 | 1296.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 12:45:00 | 1290.05 | 1290.44 | 1296.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 1288.60 | 1290.44 | 1296.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 1295.20 | 1288.08 | 1293.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 15:15:00 | 1301.50 | 1295.39 | 1294.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 15:15:00 | 1301.50 | 1295.39 | 1294.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 1306.70 | 1298.41 | 1296.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1287.90 | 1299.12 | 1297.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1287.90 | 1299.12 | 1297.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1287.90 | 1299.12 | 1297.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1296.50 | 1299.12 | 1297.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 1282.80 | 1295.86 | 1296.28 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 1299.50 | 1296.87 | 1296.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 1306.90 | 1298.88 | 1297.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 1371.80 | 1371.95 | 1353.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:30:00 | 1374.80 | 1371.95 | 1353.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1422.70 | 1436.96 | 1419.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 11:30:00 | 1460.00 | 1445.69 | 1426.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:45:00 | 1464.40 | 1451.05 | 1432.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 10:45:00 | 1455.70 | 1454.13 | 1440.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 11:45:00 | 1455.50 | 1454.31 | 1441.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1490.80 | 1496.55 | 1490.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1500.40 | 1496.55 | 1490.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 13:30:00 | 1498.20 | 1497.88 | 1493.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:30:00 | 1498.40 | 1495.83 | 1493.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 1487.40 | 1494.19 | 1493.55 | SL hit (close<static) qty=1.00 sl=1490.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 1486.10 | 1492.24 | 1492.85 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 1495.60 | 1493.57 | 1493.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 1504.00 | 1495.65 | 1494.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 14:15:00 | 1493.80 | 1495.28 | 1494.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 14:15:00 | 1493.80 | 1495.28 | 1494.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1493.80 | 1495.28 | 1494.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1493.80 | 1495.28 | 1494.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1495.10 | 1495.25 | 1494.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1504.00 | 1495.25 | 1494.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1505.10 | 1497.22 | 1495.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:15:00 | 1512.80 | 1498.19 | 1495.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 1495.80 | 1497.33 | 1497.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 12:15:00 | 1495.80 | 1497.33 | 1497.53 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1534.30 | 1503.69 | 1500.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1545.80 | 1516.59 | 1506.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 1611.10 | 1615.92 | 1598.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 1611.10 | 1615.92 | 1598.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1609.10 | 1614.19 | 1605.88 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 1598.80 | 1602.64 | 1603.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 1591.80 | 1600.47 | 1602.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1590.00 | 1587.05 | 1593.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 1590.00 | 1587.05 | 1593.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1596.50 | 1588.94 | 1593.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1596.50 | 1588.94 | 1593.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1590.20 | 1589.19 | 1593.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1565.30 | 1594.16 | 1594.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1599.60 | 1579.34 | 1583.90 | SL hit (close>static) qty=1.00 sl=1599.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1598.70 | 1586.75 | 1586.30 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 1586.70 | 1590.93 | 1591.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 1585.00 | 1589.13 | 1590.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1593.40 | 1587.25 | 1588.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1593.40 | 1587.25 | 1588.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1593.40 | 1587.25 | 1588.75 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1599.60 | 1591.32 | 1590.31 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1575.60 | 1589.88 | 1589.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1551.10 | 1573.85 | 1581.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 1557.10 | 1548.88 | 1556.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1557.10 | 1548.88 | 1556.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1557.10 | 1548.88 | 1556.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:15:00 | 1561.00 | 1548.88 | 1556.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1562.80 | 1551.67 | 1556.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:45:00 | 1563.60 | 1551.67 | 1556.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1557.60 | 1552.85 | 1557.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 1558.70 | 1552.85 | 1557.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1555.80 | 1553.44 | 1556.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1555.80 | 1553.44 | 1556.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1565.10 | 1555.77 | 1557.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 1565.40 | 1555.77 | 1557.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1556.80 | 1555.98 | 1557.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 1555.30 | 1556.78 | 1557.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 1568.20 | 1558.50 | 1558.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1568.20 | 1558.50 | 1558.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1571.50 | 1563.47 | 1560.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 1565.10 | 1565.45 | 1562.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 14:00:00 | 1565.10 | 1565.45 | 1562.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1626.00 | 1628.53 | 1613.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1619.20 | 1628.53 | 1613.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1647.00 | 1641.72 | 1629.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1654.20 | 1641.72 | 1629.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 1687.80 | 1695.27 | 1695.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1687.80 | 1695.27 | 1695.42 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1694.30 | 1689.10 | 1688.95 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 13:15:00 | 1677.90 | 1688.80 | 1689.07 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1703.70 | 1689.95 | 1689.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 1711.00 | 1696.34 | 1692.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1695.60 | 1699.92 | 1695.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1695.60 | 1699.92 | 1695.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1695.60 | 1699.92 | 1695.73 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 1684.10 | 1692.33 | 1693.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1663.30 | 1680.00 | 1685.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 1684.00 | 1677.49 | 1682.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 1684.00 | 1677.49 | 1682.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1684.00 | 1677.49 | 1682.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 1684.00 | 1677.49 | 1682.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1687.50 | 1679.49 | 1682.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 1687.50 | 1679.49 | 1682.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1685.40 | 1680.67 | 1683.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1691.50 | 1680.67 | 1683.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1676.70 | 1679.88 | 1682.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 1688.80 | 1679.88 | 1682.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1682.00 | 1680.30 | 1682.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1680.00 | 1680.30 | 1682.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1672.90 | 1678.82 | 1681.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:45:00 | 1671.40 | 1676.23 | 1679.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1683.20 | 1675.93 | 1678.67 | SL hit (close>static) qty=1.00 sl=1682.30 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1689.00 | 1679.01 | 1678.72 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 1676.50 | 1678.52 | 1678.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1670.50 | 1676.92 | 1677.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 1635.40 | 1631.18 | 1642.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:00:00 | 1635.40 | 1631.18 | 1642.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1639.70 | 1632.88 | 1642.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 1643.30 | 1632.88 | 1642.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1603.60 | 1595.09 | 1604.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 1603.60 | 1595.09 | 1604.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1598.30 | 1595.73 | 1603.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 1583.80 | 1595.73 | 1603.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:00:00 | 1595.30 | 1584.18 | 1585.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 1603.30 | 1589.70 | 1588.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1603.30 | 1589.70 | 1588.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 14:15:00 | 1609.40 | 1593.64 | 1590.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1583.20 | 1593.93 | 1590.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1583.20 | 1593.93 | 1590.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1583.20 | 1593.93 | 1590.95 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1566.60 | 1584.76 | 1587.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 1563.70 | 1578.83 | 1583.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 1546.60 | 1545.34 | 1552.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:45:00 | 1547.70 | 1545.34 | 1552.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1539.40 | 1544.99 | 1549.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1530.50 | 1541.66 | 1546.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 1453.97 | 1474.24 | 1495.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 1456.50 | 1451.58 | 1466.03 | SL hit (close>ema200) qty=0.50 sl=1451.58 alert=retest2 |

### Cycle 172 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1472.60 | 1462.94 | 1462.29 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1451.80 | 1461.00 | 1461.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 1441.00 | 1457.00 | 1459.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 1449.20 | 1443.10 | 1449.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 10:15:00 | 1449.20 | 1443.10 | 1449.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1449.20 | 1443.10 | 1449.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 1449.20 | 1443.10 | 1449.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1446.40 | 1443.76 | 1449.40 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1474.70 | 1456.40 | 1454.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 1477.70 | 1463.64 | 1458.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1468.10 | 1475.66 | 1468.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1468.10 | 1475.66 | 1468.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1468.10 | 1475.66 | 1468.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1468.10 | 1475.66 | 1468.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1456.80 | 1471.89 | 1467.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1456.80 | 1471.89 | 1467.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1471.30 | 1471.77 | 1467.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 1454.40 | 1471.77 | 1467.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1466.30 | 1470.52 | 1467.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 1466.30 | 1470.52 | 1467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1458.70 | 1468.16 | 1467.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 1458.70 | 1468.16 | 1467.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 1456.70 | 1465.87 | 1466.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1451.80 | 1462.58 | 1464.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1463.50 | 1461.10 | 1463.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 1463.50 | 1461.10 | 1463.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1463.50 | 1461.10 | 1463.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1463.50 | 1461.10 | 1463.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1482.10 | 1465.30 | 1464.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 1488.10 | 1469.86 | 1467.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 14:15:00 | 1479.70 | 1479.94 | 1474.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 15:00:00 | 1479.70 | 1479.94 | 1474.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1474.80 | 1479.31 | 1475.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:45:00 | 1473.10 | 1479.31 | 1475.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1481.30 | 1479.71 | 1475.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1495.30 | 1480.68 | 1477.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 1487.40 | 1499.92 | 1500.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1487.40 | 1499.92 | 1500.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 1471.20 | 1489.82 | 1494.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 1484.60 | 1479.24 | 1485.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 12:00:00 | 1484.60 | 1479.24 | 1485.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1491.60 | 1481.71 | 1486.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:45:00 | 1491.10 | 1481.71 | 1486.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1491.50 | 1483.67 | 1486.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 1493.50 | 1483.67 | 1486.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 1499.60 | 1488.95 | 1488.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1507.10 | 1492.58 | 1490.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 1520.60 | 1521.34 | 1513.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 1512.30 | 1521.34 | 1513.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1505.90 | 1518.25 | 1512.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 1506.90 | 1518.25 | 1512.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1499.80 | 1514.56 | 1511.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1499.70 | 1514.56 | 1511.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1511.30 | 1512.81 | 1511.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:15:00 | 1512.70 | 1512.81 | 1511.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1505.10 | 1511.19 | 1510.84 | SL hit (close<static) qty=1.00 sl=1506.10 alert=retest2 |

### Cycle 179 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1503.00 | 1509.55 | 1510.12 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 1527.90 | 1513.22 | 1511.74 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 1500.90 | 1515.16 | 1516.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1484.30 | 1507.30 | 1512.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 1503.40 | 1502.94 | 1508.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 1503.40 | 1502.94 | 1508.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1503.40 | 1502.94 | 1508.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:45:00 | 1505.70 | 1502.94 | 1508.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1502.00 | 1498.46 | 1503.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 1501.10 | 1498.46 | 1503.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1500.10 | 1492.07 | 1497.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1504.40 | 1492.07 | 1497.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1504.40 | 1494.54 | 1498.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1504.40 | 1494.54 | 1498.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1504.90 | 1496.61 | 1499.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 1504.90 | 1496.61 | 1499.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1506.90 | 1501.56 | 1500.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1516.50 | 1505.16 | 1502.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 1509.10 | 1511.10 | 1507.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 09:15:00 | 1515.20 | 1511.10 | 1507.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1508.70 | 1510.62 | 1507.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 1508.70 | 1510.62 | 1507.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1503.00 | 1509.10 | 1507.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1503.00 | 1509.10 | 1507.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1508.00 | 1508.88 | 1507.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 1502.60 | 1508.88 | 1507.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1501.90 | 1507.48 | 1506.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1501.90 | 1507.48 | 1506.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1505.90 | 1507.17 | 1506.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1509.00 | 1507.17 | 1506.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1499.50 | 1506.11 | 1506.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 1499.50 | 1506.11 | 1506.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 1496.90 | 1502.06 | 1504.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1486.60 | 1472.11 | 1479.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1486.60 | 1472.11 | 1479.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1486.60 | 1472.11 | 1479.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1490.70 | 1472.11 | 1479.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1489.60 | 1475.61 | 1480.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1489.90 | 1475.61 | 1480.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1494.70 | 1483.60 | 1483.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1528.50 | 1496.60 | 1489.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1507.70 | 1517.10 | 1506.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 1507.70 | 1517.10 | 1506.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1513.50 | 1516.38 | 1507.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 1510.20 | 1516.38 | 1507.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1510.80 | 1520.75 | 1517.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1510.80 | 1520.75 | 1517.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1511.40 | 1518.88 | 1516.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 1509.40 | 1518.88 | 1516.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1519.70 | 1519.10 | 1517.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1519.60 | 1519.10 | 1517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1517.10 | 1519.98 | 1518.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 1517.30 | 1519.98 | 1518.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1516.60 | 1519.31 | 1518.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 1516.60 | 1519.31 | 1518.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1522.10 | 1519.86 | 1518.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 1525.20 | 1520.29 | 1518.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1498.10 | 1540.21 | 1541.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1498.10 | 1540.21 | 1541.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1447.90 | 1478.11 | 1497.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 1410.40 | 1407.30 | 1423.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:45:00 | 1412.50 | 1407.30 | 1423.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1422.10 | 1411.84 | 1421.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1422.10 | 1411.84 | 1421.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1422.80 | 1414.03 | 1421.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:15:00 | 1424.30 | 1414.03 | 1421.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1417.70 | 1414.77 | 1421.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1425.10 | 1414.77 | 1421.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1409.30 | 1413.67 | 1420.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:15:00 | 1405.50 | 1413.67 | 1420.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 1420.50 | 1412.13 | 1416.30 | SL hit (close>static) qty=1.00 sl=1420.40 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1424.60 | 1413.09 | 1412.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 1432.00 | 1416.87 | 1414.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 1428.60 | 1429.95 | 1423.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 1428.60 | 1429.95 | 1423.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1467.20 | 1463.89 | 1455.03 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1441.20 | 1454.50 | 1454.70 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 1468.80 | 1455.53 | 1454.52 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 1450.20 | 1457.51 | 1458.37 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1463.00 | 1459.44 | 1459.09 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1450.10 | 1458.34 | 1458.77 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1484.30 | 1454.84 | 1452.99 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 1453.60 | 1459.66 | 1459.80 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 10:15:00 | 1465.90 | 1460.86 | 1460.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 1473.60 | 1463.40 | 1461.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 14:15:00 | 1462.40 | 1464.58 | 1462.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 14:15:00 | 1462.40 | 1464.58 | 1462.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1462.40 | 1464.58 | 1462.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1462.40 | 1464.58 | 1462.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1464.00 | 1464.47 | 1462.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1465.00 | 1464.47 | 1462.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1455.60 | 1462.69 | 1462.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1455.60 | 1462.69 | 1462.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1448.30 | 1459.82 | 1460.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1443.30 | 1456.51 | 1459.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 1450.20 | 1449.05 | 1453.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 10:30:00 | 1448.30 | 1449.05 | 1453.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1452.60 | 1449.76 | 1453.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 1452.60 | 1449.76 | 1453.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1455.60 | 1450.93 | 1453.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:45:00 | 1455.80 | 1450.93 | 1453.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1450.90 | 1450.92 | 1453.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1449.50 | 1451.88 | 1453.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1407.00 | 1402.87 | 1402.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1407.00 | 1402.87 | 1402.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1427.40 | 1408.41 | 1405.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1436.50 | 1447.12 | 1438.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1436.50 | 1447.12 | 1438.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1436.50 | 1447.12 | 1438.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1436.50 | 1447.12 | 1438.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1444.20 | 1446.54 | 1439.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1435.30 | 1446.54 | 1439.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1436.10 | 1444.45 | 1439.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 1436.10 | 1444.45 | 1439.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1424.50 | 1440.46 | 1437.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 1424.50 | 1440.46 | 1437.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1427.50 | 1437.87 | 1436.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:15:00 | 1429.20 | 1437.87 | 1436.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 1440.50 | 1438.39 | 1437.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 1433.00 | 1440.62 | 1440.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1433.00 | 1440.62 | 1440.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 1425.60 | 1437.61 | 1439.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 1436.10 | 1431.32 | 1435.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 1436.10 | 1431.32 | 1435.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1436.10 | 1431.32 | 1435.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1436.10 | 1431.32 | 1435.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1437.00 | 1432.46 | 1435.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1441.10 | 1432.46 | 1435.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1441.80 | 1434.33 | 1436.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1441.80 | 1434.33 | 1436.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1442.40 | 1435.94 | 1436.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:45:00 | 1442.00 | 1435.94 | 1436.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 1439.10 | 1437.41 | 1437.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 1452.80 | 1440.41 | 1438.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1442.00 | 1450.64 | 1446.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1442.00 | 1450.64 | 1446.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1442.00 | 1450.64 | 1446.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1442.00 | 1450.64 | 1446.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1453.60 | 1451.23 | 1446.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 11:15:00 | 1459.60 | 1451.23 | 1446.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:00:00 | 1455.90 | 1454.46 | 1449.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 15:15:00 | 1561.50 | 1570.12 | 1570.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 15:15:00 | 1561.50 | 1570.12 | 1570.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 1558.80 | 1565.64 | 1567.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 1562.20 | 1558.96 | 1563.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 1562.20 | 1558.96 | 1563.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1562.20 | 1558.96 | 1563.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1562.20 | 1558.96 | 1563.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1558.60 | 1558.89 | 1562.67 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1573.30 | 1564.91 | 1564.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1577.20 | 1570.02 | 1567.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 1574.00 | 1578.15 | 1573.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 13:15:00 | 1574.00 | 1578.15 | 1573.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1574.00 | 1578.15 | 1573.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 1574.00 | 1578.15 | 1573.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1576.00 | 1577.72 | 1573.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 1579.00 | 1574.79 | 1573.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 1577.90 | 1574.91 | 1573.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 1577.80 | 1574.91 | 1573.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 1579.30 | 1576.08 | 1574.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1575.00 | 1575.86 | 1574.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1582.30 | 1575.86 | 1574.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1578.20 | 1576.33 | 1574.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1587.00 | 1577.41 | 1576.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 1588.30 | 1579.02 | 1577.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 1613.30 | 1626.55 | 1626.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1613.30 | 1626.55 | 1626.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1612.00 | 1619.58 | 1623.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1626.40 | 1618.96 | 1621.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1626.40 | 1618.96 | 1621.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1626.40 | 1618.96 | 1621.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 1625.90 | 1618.96 | 1621.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1619.60 | 1619.08 | 1621.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 1615.10 | 1619.08 | 1621.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 1617.30 | 1619.15 | 1621.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1610.70 | 1605.36 | 1605.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1610.70 | 1605.36 | 1605.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1615.50 | 1607.39 | 1606.08 | Break + close above crossover candle high |

### Cycle 203 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1590.20 | 1606.68 | 1606.74 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 1606.60 | 1604.65 | 1604.41 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 13:15:00 | 1598.10 | 1603.34 | 1603.84 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1616.90 | 1605.15 | 1604.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 1629.00 | 1609.92 | 1606.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1609.60 | 1618.63 | 1613.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1609.60 | 1618.63 | 1613.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1609.60 | 1618.63 | 1613.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1616.80 | 1618.63 | 1613.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1588.90 | 1612.68 | 1611.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1588.90 | 1612.68 | 1611.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1592.50 | 1608.65 | 1609.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1583.30 | 1600.77 | 1605.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1583.70 | 1579.99 | 1588.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1583.70 | 1579.99 | 1588.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1585.70 | 1581.13 | 1587.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1593.80 | 1581.13 | 1587.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1583.50 | 1581.61 | 1587.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:30:00 | 1585.60 | 1581.61 | 1587.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1585.10 | 1582.91 | 1587.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1605.80 | 1582.91 | 1587.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1595.50 | 1585.43 | 1587.79 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1614.80 | 1591.30 | 1590.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 1617.70 | 1602.44 | 1596.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 11:15:00 | 1599.30 | 1603.34 | 1598.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 1599.30 | 1603.34 | 1598.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1599.30 | 1603.34 | 1598.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 1599.90 | 1603.34 | 1598.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1593.10 | 1601.29 | 1598.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 1593.10 | 1601.29 | 1598.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1586.70 | 1598.38 | 1597.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1586.70 | 1598.38 | 1597.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 14:15:00 | 1587.30 | 1596.16 | 1596.40 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1639.00 | 1604.09 | 1599.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1647.80 | 1612.84 | 1604.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1692.80 | 1705.13 | 1676.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 10:00:00 | 1692.80 | 1705.13 | 1676.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1678.80 | 1694.01 | 1678.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1678.80 | 1694.01 | 1678.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 1677.40 | 1690.69 | 1678.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 1677.40 | 1690.69 | 1678.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1677.40 | 1688.03 | 1678.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 1677.40 | 1688.03 | 1678.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1675.70 | 1685.56 | 1677.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1680.60 | 1685.56 | 1677.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1678.70 | 1684.19 | 1677.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 1701.40 | 1681.85 | 1678.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 1698.80 | 1690.71 | 1684.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 1695.00 | 1689.26 | 1685.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 11:00:00 | 1699.50 | 1693.65 | 1688.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1738.80 | 1755.96 | 1746.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1738.80 | 1755.96 | 1746.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1744.00 | 1753.57 | 1745.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 1747.00 | 1746.44 | 1744.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 1732.00 | 1741.81 | 1742.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 1732.00 | 1741.81 | 1742.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1718.30 | 1737.03 | 1740.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 1723.80 | 1723.51 | 1729.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 1751.60 | 1723.51 | 1729.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1736.90 | 1726.19 | 1730.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:00:00 | 1729.80 | 1731.60 | 1732.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 09:15:00 | 1643.31 | 1706.23 | 1719.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 14:15:00 | 1645.10 | 1644.58 | 1663.91 | SL hit (close>ema200) qty=0.50 sl=1644.58 alert=retest2 |

### Cycle 212 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 1647.10 | 1636.18 | 1635.91 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1625.10 | 1636.35 | 1637.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1572.80 | 1623.64 | 1631.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 1542.10 | 1541.52 | 1569.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 1542.10 | 1541.52 | 1569.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1530.70 | 1518.89 | 1536.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1535.60 | 1518.89 | 1536.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1541.50 | 1524.79 | 1536.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 1542.00 | 1524.79 | 1536.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1542.70 | 1528.37 | 1536.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:15:00 | 1537.30 | 1528.37 | 1536.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1460.43 | 1485.86 | 1499.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 10:15:00 | 1383.57 | 1424.67 | 1449.73 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 214 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1338.90 | 1332.72 | 1332.64 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1327.90 | 1338.26 | 1338.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1321.80 | 1334.97 | 1337.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 1336.80 | 1334.19 | 1335.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 1336.80 | 1334.19 | 1335.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1336.80 | 1334.19 | 1335.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 1339.60 | 1334.19 | 1335.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1324.10 | 1332.17 | 1334.86 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1349.30 | 1337.40 | 1336.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 11:15:00 | 1350.90 | 1340.10 | 1337.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 14:15:00 | 1344.00 | 1344.42 | 1340.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 1344.00 | 1344.42 | 1340.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1339.70 | 1343.47 | 1340.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 1366.30 | 1343.47 | 1340.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 1332.80 | 1354.92 | 1357.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1332.80 | 1354.92 | 1357.26 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1382.30 | 1359.61 | 1358.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 1387.80 | 1365.25 | 1361.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 14:15:00 | 1385.00 | 1390.36 | 1381.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 14:15:00 | 1385.00 | 1390.36 | 1381.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 1385.00 | 1390.36 | 1381.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 1385.00 | 1390.36 | 1381.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1389.00 | 1390.09 | 1381.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1412.30 | 1390.09 | 1381.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 13:15:00 | 1391.00 | 1405.05 | 1406.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1391.00 | 1405.05 | 1406.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1389.40 | 1401.92 | 1404.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 1401.80 | 1387.42 | 1393.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 1401.80 | 1387.42 | 1393.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1401.80 | 1387.42 | 1393.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 1408.20 | 1387.42 | 1393.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1416.10 | 1393.16 | 1395.48 | EMA400 retest candle locked (from downside) |

### Cycle 220 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1420.30 | 1401.31 | 1398.95 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1393.60 | 1399.22 | 1399.32 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 1400.60 | 1399.49 | 1399.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1418.30 | 1403.26 | 1401.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 09:15:00 | 1458.30 | 1466.55 | 1452.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1458.30 | 1466.55 | 1452.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1458.30 | 1466.55 | 1452.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1458.00 | 1466.55 | 1452.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1451.50 | 1463.54 | 1452.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 1451.50 | 1463.54 | 1452.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1452.90 | 1461.41 | 1452.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:45:00 | 1453.80 | 1461.41 | 1452.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 1451.80 | 1459.49 | 1452.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 1451.80 | 1459.49 | 1452.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 1448.50 | 1457.29 | 1452.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:00:00 | 1448.50 | 1457.29 | 1452.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 1450.60 | 1455.95 | 1452.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:45:00 | 1449.30 | 1455.95 | 1452.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1437.50 | 1451.95 | 1450.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 1456.10 | 1452.44 | 1451.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 1455.40 | 1453.25 | 1451.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 1431.00 | 1447.97 | 1449.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 10:15:00 | 1431.00 | 1447.97 | 1449.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1423.30 | 1437.09 | 1442.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 1438.00 | 1435.72 | 1439.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 09:15:00 | 1467.90 | 1435.72 | 1439.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1462.10 | 1441.00 | 1441.52 | EMA400 retest candle locked (from downside) |

### Cycle 224 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1462.40 | 1445.28 | 1443.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1467.00 | 1449.62 | 1445.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 1504.00 | 1505.89 | 1492.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:45:00 | 1501.40 | 1505.89 | 1492.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1497.50 | 1506.76 | 1499.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 1499.20 | 1506.76 | 1499.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1509.00 | 1507.21 | 1500.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 1498.30 | 1507.21 | 1500.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1508.10 | 1507.39 | 1500.99 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1449.70 | 1491.06 | 1495.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 1422.60 | 1477.37 | 1488.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 1471.80 | 1457.70 | 1475.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 1471.80 | 1457.70 | 1475.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 1471.80 | 1457.70 | 1475.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 1471.80 | 1457.70 | 1475.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1460.10 | 1458.18 | 1473.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:30:00 | 1466.50 | 1458.18 | 1473.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1425.00 | 1452.64 | 1468.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:15:00 | 1416.90 | 1438.58 | 1455.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 1412.50 | 1403.36 | 1403.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 15:15:00 | 1412.50 | 1403.36 | 1403.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1452.70 | 1413.23 | 1407.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 10:15:00 | 1465.00 | 1466.20 | 1452.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:45:00 | 1459.10 | 1466.20 | 1452.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 1462.60 | 1467.73 | 1459.20 | EMA400 retest candle locked (from upside) |

### Cycle 227 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1451.20 | 1456.24 | 1456.26 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1473.60 | 1459.71 | 1457.84 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 1454.40 | 1458.75 | 1459.00 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 1463.50 | 1458.96 | 1458.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 13:15:00 | 1465.50 | 1460.27 | 1459.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 1460.90 | 1461.06 | 1459.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 15:15:00 | 1460.90 | 1461.06 | 1459.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1460.90 | 1461.06 | 1459.66 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 10:45:00 | 1061.80 | 2023-05-17 11:15:00 | 1044.55 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-05-26 09:15:00 | 1111.30 | 2023-05-30 13:15:00 | 1103.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-05-30 12:00:00 | 1103.75 | 2023-05-30 13:15:00 | 1103.10 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-06-08 12:00:00 | 1085.85 | 2023-06-13 10:15:00 | 1084.55 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2023-06-23 12:00:00 | 1115.15 | 2023-06-28 15:15:00 | 1108.25 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-23 14:15:00 | 1113.75 | 2023-06-28 15:15:00 | 1108.25 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-26 10:00:00 | 1115.00 | 2023-06-28 15:15:00 | 1108.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-06-26 11:00:00 | 1115.25 | 2023-06-28 15:15:00 | 1108.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-07-13 10:15:00 | 1188.50 | 2023-07-21 09:15:00 | 1205.75 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2023-07-13 11:45:00 | 1188.75 | 2023-07-21 09:15:00 | 1205.75 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2023-07-14 09:15:00 | 1188.85 | 2023-07-21 09:15:00 | 1205.75 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2023-08-02 12:30:00 | 1137.70 | 2023-08-14 11:15:00 | 1216.10 | STOP_HIT | 1.00 | 6.89% |
| BUY | retest2 | 2023-08-02 14:45:00 | 1137.25 | 2023-08-14 11:15:00 | 1216.10 | STOP_HIT | 1.00 | 6.93% |
| BUY | retest2 | 2023-08-03 14:45:00 | 1139.50 | 2023-08-14 11:15:00 | 1216.10 | STOP_HIT | 1.00 | 6.72% |
| BUY | retest2 | 2023-08-17 13:30:00 | 1225.35 | 2023-08-18 09:15:00 | 1213.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-08-24 10:45:00 | 1205.10 | 2023-08-29 12:15:00 | 1201.65 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2023-08-24 12:00:00 | 1204.05 | 2023-08-29 12:15:00 | 1201.65 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2023-08-24 13:15:00 | 1201.45 | 2023-08-29 12:15:00 | 1201.65 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2023-08-24 14:45:00 | 1204.10 | 2023-08-29 12:15:00 | 1201.65 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2023-08-31 09:15:00 | 1205.85 | 2023-09-13 10:15:00 | 1254.55 | STOP_HIT | 1.00 | 4.04% |
| SELL | retest2 | 2023-09-28 09:15:00 | 1265.50 | 2023-10-04 09:15:00 | 1202.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-28 09:15:00 | 1265.50 | 2023-10-04 15:15:00 | 1206.60 | STOP_HIT | 0.50 | 4.65% |
| BUY | retest2 | 2023-10-11 14:30:00 | 1234.55 | 2023-10-12 09:15:00 | 1220.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-10-18 11:45:00 | 1184.50 | 2023-10-26 09:15:00 | 1125.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 14:30:00 | 1186.45 | 2023-10-26 09:15:00 | 1127.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 11:45:00 | 1184.50 | 2023-10-27 09:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2023-10-18 14:30:00 | 1186.45 | 2023-10-27 09:15:00 | 1123.40 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2023-11-02 14:30:00 | 1117.95 | 2023-11-03 09:15:00 | 1132.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest1 | 2023-11-20 09:15:00 | 1208.95 | 2023-11-21 15:15:00 | 1202.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2023-11-21 11:15:00 | 1206.35 | 2023-11-21 15:15:00 | 1202.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-11-22 09:15:00 | 1210.70 | 2023-11-24 12:15:00 | 1204.30 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-11-24 11:15:00 | 1207.80 | 2023-11-24 12:15:00 | 1204.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2023-12-20 09:15:00 | 1294.80 | 2023-12-20 12:15:00 | 1273.35 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-12-29 09:45:00 | 1288.55 | 2023-12-29 14:15:00 | 1272.15 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-01-08 11:45:00 | 1231.90 | 2024-01-11 09:15:00 | 1253.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-01-10 09:15:00 | 1230.20 | 2024-01-11 09:15:00 | 1253.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-01-17 14:00:00 | 1329.15 | 2024-01-25 10:15:00 | 1327.75 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-01-17 15:15:00 | 1333.00 | 2024-01-25 10:15:00 | 1327.75 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-01-18 10:30:00 | 1340.70 | 2024-01-25 10:15:00 | 1327.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-01-30 13:45:00 | 1322.30 | 2024-02-02 10:15:00 | 1342.70 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-01-31 10:15:00 | 1324.70 | 2024-02-02 10:15:00 | 1342.70 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-02-01 09:15:00 | 1320.35 | 2024-02-02 10:15:00 | 1342.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-02-01 11:00:00 | 1323.85 | 2024-02-02 10:15:00 | 1342.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest1 | 2024-02-09 12:15:00 | 1307.20 | 2024-02-12 09:15:00 | 1318.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-02-13 09:15:00 | 1309.50 | 2024-02-13 14:15:00 | 1327.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-02-13 10:15:00 | 1313.00 | 2024-02-13 14:15:00 | 1327.95 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-02-13 12:00:00 | 1311.55 | 2024-02-13 14:15:00 | 1327.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-02-16 12:00:00 | 1299.80 | 2024-02-19 11:15:00 | 1314.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-02-16 13:45:00 | 1298.70 | 2024-02-19 11:15:00 | 1314.25 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-02-19 09:15:00 | 1292.90 | 2024-02-19 11:15:00 | 1314.25 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-02-28 13:00:00 | 1290.60 | 2024-03-07 11:15:00 | 1280.25 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-02-28 14:30:00 | 1291.40 | 2024-03-07 11:15:00 | 1280.25 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-02-28 15:00:00 | 1288.80 | 2024-03-07 11:15:00 | 1280.25 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2024-03-18 10:30:00 | 1274.55 | 2024-03-21 10:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-03-18 15:00:00 | 1272.90 | 2024-03-21 10:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-03-22 14:00:00 | 1275.25 | 2024-03-22 14:15:00 | 1259.10 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-04-02 10:45:00 | 1238.30 | 2024-04-03 09:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-04-02 14:15:00 | 1238.50 | 2024-04-03 09:15:00 | 1260.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-04-09 09:15:00 | 1274.60 | 2024-04-09 11:15:00 | 1256.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-05-27 09:30:00 | 1325.30 | 2024-05-28 09:15:00 | 1320.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1218.70 | 2024-06-05 10:15:00 | 1266.35 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1234.05 | 2024-06-05 10:15:00 | 1266.35 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-06-04 11:45:00 | 1224.80 | 2024-06-05 10:15:00 | 1266.35 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-06-04 14:00:00 | 1235.20 | 2024-06-05 10:15:00 | 1266.35 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-06-18 09:15:00 | 1383.15 | 2024-06-19 09:15:00 | 1362.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-06-18 13:00:00 | 1378.35 | 2024-06-19 09:15:00 | 1362.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-06-24 11:00:00 | 1396.40 | 2024-07-04 14:15:00 | 1458.45 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest2 | 2024-06-24 11:45:00 | 1403.65 | 2024-07-04 14:15:00 | 1458.45 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2024-06-27 09:45:00 | 1396.00 | 2024-07-04 14:15:00 | 1458.45 | STOP_HIT | 1.00 | 4.47% |
| SELL | retest2 | 2024-07-11 11:00:00 | 1448.75 | 2024-07-12 09:15:00 | 1480.10 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-07-18 09:45:00 | 1519.60 | 2024-07-19 13:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-07-18 10:15:00 | 1520.50 | 2024-07-19 13:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-18 12:15:00 | 1519.65 | 2024-07-19 13:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-07-25 12:15:00 | 1535.05 | 2024-07-26 09:15:00 | 1500.10 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-07-25 14:00:00 | 1535.25 | 2024-07-26 09:15:00 | 1500.10 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-07-25 15:15:00 | 1540.00 | 2024-07-26 09:15:00 | 1500.10 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-07-26 12:00:00 | 1538.20 | 2024-07-29 13:15:00 | 1521.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-06 11:30:00 | 1484.80 | 2024-08-07 09:15:00 | 1501.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1482.55 | 2024-08-07 09:15:00 | 1501.55 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-08-06 14:30:00 | 1482.20 | 2024-08-07 09:15:00 | 1501.55 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-06 15:00:00 | 1482.70 | 2024-08-07 09:15:00 | 1501.55 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-08-07 12:30:00 | 1489.45 | 2024-08-09 11:15:00 | 1500.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1518.85 | 2024-08-23 12:15:00 | 1597.35 | STOP_HIT | 1.00 | 5.17% |
| BUY | retest2 | 2024-08-29 12:00:00 | 1643.00 | 2024-09-04 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-08-29 13:00:00 | 1644.80 | 2024-09-04 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-08-29 14:45:00 | 1652.95 | 2024-09-04 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-09-02 09:15:00 | 1650.20 | 2024-09-04 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-09-02 12:00:00 | 1658.85 | 2024-09-04 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-09-06 13:30:00 | 1623.05 | 2024-09-11 12:15:00 | 1623.15 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-09-09 09:15:00 | 1612.00 | 2024-09-11 12:15:00 | 1623.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-09 11:15:00 | 1622.50 | 2024-09-11 12:15:00 | 1623.15 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-09-17 10:15:00 | 1657.10 | 2024-09-18 09:15:00 | 1600.85 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-09-17 14:15:00 | 1657.20 | 2024-09-18 09:15:00 | 1600.85 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2024-09-24 09:15:00 | 1602.90 | 2024-09-24 13:15:00 | 1631.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-09-24 11:30:00 | 1603.85 | 2024-09-24 13:15:00 | 1631.55 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-09-26 13:30:00 | 1597.25 | 2024-09-27 09:15:00 | 1626.95 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1622.60 | 2024-10-07 10:15:00 | 1609.45 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-10-07 11:45:00 | 1620.00 | 2024-10-08 09:15:00 | 1610.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-10-07 13:00:00 | 1619.55 | 2024-10-08 09:15:00 | 1610.45 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-10-07 13:45:00 | 1621.40 | 2024-10-08 09:15:00 | 1610.45 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-10-08 13:45:00 | 1630.40 | 2024-10-10 15:15:00 | 1612.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-10-08 15:00:00 | 1630.00 | 2024-10-10 15:15:00 | 1612.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-10-09 09:15:00 | 1660.65 | 2024-10-10 15:15:00 | 1612.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-10-10 14:15:00 | 1633.70 | 2024-10-10 15:15:00 | 1612.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-10-17 09:45:00 | 1685.70 | 2024-10-18 09:15:00 | 1655.25 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-10-17 10:30:00 | 1677.90 | 2024-10-18 09:15:00 | 1655.25 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-10-17 12:30:00 | 1684.85 | 2024-10-18 09:15:00 | 1655.25 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1683.50 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1742.70 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-10-21 11:45:00 | 1706.00 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2024-10-21 12:30:00 | 1702.50 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2024-10-21 13:45:00 | 1707.00 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-10-23 10:15:00 | 1724.00 | 2024-10-25 15:15:00 | 1711.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-30 14:00:00 | 1683.85 | 2024-10-31 10:15:00 | 1599.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-30 15:00:00 | 1683.30 | 2024-10-31 10:15:00 | 1599.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-30 14:00:00 | 1683.85 | 2024-11-04 09:15:00 | 1620.25 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2024-10-30 15:00:00 | 1683.30 | 2024-11-04 09:15:00 | 1620.25 | STOP_HIT | 0.50 | 3.75% |
| BUY | retest2 | 2024-11-08 09:15:00 | 1665.95 | 2024-11-13 10:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest1 | 2024-11-21 09:15:00 | 1707.70 | 2024-11-21 11:15:00 | 1689.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest1 | 2024-11-21 10:00:00 | 1700.80 | 2024-11-21 11:15:00 | 1689.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-11-21 15:00:00 | 1702.75 | 2024-11-28 11:15:00 | 1716.05 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2024-12-11 12:30:00 | 1754.90 | 2024-12-12 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-12-11 14:15:00 | 1754.20 | 2024-12-12 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-12-11 15:15:00 | 1755.15 | 2024-12-12 09:15:00 | 1800.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-12-24 14:00:00 | 1704.80 | 2024-12-30 10:15:00 | 1714.95 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1705.70 | 2024-12-30 10:15:00 | 1714.95 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-27 11:45:00 | 1705.80 | 2024-12-30 10:15:00 | 1714.95 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1704.30 | 2024-12-30 10:15:00 | 1714.95 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1654.40 | 2025-01-10 09:15:00 | 1676.05 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-09 10:45:00 | 1651.80 | 2025-01-10 09:15:00 | 1676.05 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-02-03 15:15:00 | 1651.05 | 2025-02-05 14:15:00 | 1661.35 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-02-04 10:30:00 | 1650.75 | 2025-02-05 14:15:00 | 1661.35 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-02-04 11:30:00 | 1652.25 | 2025-02-05 14:15:00 | 1661.35 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-02-05 09:45:00 | 1652.40 | 2025-02-05 14:15:00 | 1661.35 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-02-07 12:15:00 | 1679.60 | 2025-02-11 15:15:00 | 1671.15 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1446.00 | 2025-03-24 10:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-03-18 10:45:00 | 1442.80 | 2025-03-24 10:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-03-28 09:15:00 | 1417.35 | 2025-04-02 13:15:00 | 1421.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-03-28 11:00:00 | 1420.90 | 2025-04-02 13:15:00 | 1421.65 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-03-28 13:00:00 | 1419.40 | 2025-04-02 13:15:00 | 1421.65 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-03-28 15:00:00 | 1419.80 | 2025-04-02 13:15:00 | 1421.65 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1275.45 | 2025-04-15 15:15:00 | 1301.50 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-11 11:45:00 | 1292.05 | 2025-04-15 15:15:00 | 1301.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-04-11 12:45:00 | 1290.05 | 2025-04-15 15:15:00 | 1301.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-11 13:15:00 | 1288.60 | 2025-04-15 15:15:00 | 1301.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-04-25 11:30:00 | 1460.00 | 2025-05-06 12:15:00 | 1487.40 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-04-25 13:45:00 | 1464.40 | 2025-05-06 12:15:00 | 1487.40 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-04-28 10:45:00 | 1455.70 | 2025-05-06 12:15:00 | 1487.40 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2025-04-28 11:45:00 | 1455.50 | 2025-05-07 09:15:00 | 1486.10 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-05-05 09:15:00 | 1500.40 | 2025-05-07 09:15:00 | 1486.10 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-05 13:30:00 | 1498.20 | 2025-05-07 09:15:00 | 1486.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-05-06 10:30:00 | 1498.40 | 2025-05-07 09:15:00 | 1486.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-05-08 11:15:00 | 1512.80 | 2025-05-09 12:15:00 | 1495.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1565.30 | 2025-05-23 09:15:00 | 1599.60 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-05-23 12:30:00 | 1588.20 | 2025-05-26 09:15:00 | 1598.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-05-23 13:15:00 | 1587.40 | 2025-05-26 09:15:00 | 1598.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-05 10:00:00 | 1555.30 | 2025-06-05 11:15:00 | 1568.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1654.20 | 2025-06-19 11:15:00 | 1687.80 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2025-07-01 13:45:00 | 1671.40 | 2025-07-02 09:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-02 10:30:00 | 1669.40 | 2025-07-03 09:15:00 | 1689.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-14 09:15:00 | 1583.80 | 2025-07-16 13:15:00 | 1603.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-16 12:00:00 | 1595.30 | 2025-07-16 13:15:00 | 1603.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1530.50 | 2025-07-28 09:15:00 | 1453.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1530.50 | 2025-07-29 13:15:00 | 1456.50 | STOP_HIT | 0.50 | 4.84% |
| BUY | retest2 | 2025-08-12 09:15:00 | 1495.30 | 2025-08-14 14:15:00 | 1487.40 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-08-22 13:15:00 | 1512.70 | 2025-08-22 14:15:00 | 1505.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-09-03 14:15:00 | 1509.00 | 2025-09-04 09:15:00 | 1499.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-09-16 14:15:00 | 1525.20 | 2025-09-22 09:15:00 | 1498.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-09-30 14:15:00 | 1405.50 | 2025-10-01 11:15:00 | 1420.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-03 11:30:00 | 1407.00 | 2025-10-06 10:15:00 | 1427.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1449.50 | 2025-11-11 14:15:00 | 1407.00 | STOP_HIT | 1.00 | 2.93% |
| BUY | retest2 | 2025-11-14 14:15:00 | 1429.20 | 2025-11-18 11:15:00 | 1433.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-11-14 15:00:00 | 1440.50 | 2025-11-18 11:15:00 | 1433.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-21 11:15:00 | 1459.60 | 2025-12-09 15:15:00 | 1561.50 | STOP_HIT | 1.00 | 6.98% |
| BUY | retest2 | 2025-11-21 13:00:00 | 1455.90 | 2025-12-09 15:15:00 | 1561.50 | STOP_HIT | 1.00 | 7.25% |
| BUY | retest2 | 2025-12-16 12:15:00 | 1579.00 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2025-12-16 12:45:00 | 1577.90 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-12-16 13:15:00 | 1577.80 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2025-12-16 15:00:00 | 1579.30 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-12-18 09:15:00 | 1587.00 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2025-12-18 11:15:00 | 1588.30 | 2025-12-26 10:15:00 | 1613.30 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2025-12-29 11:15:00 | 1615.10 | 2026-01-02 10:15:00 | 1610.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-12-29 12:15:00 | 1617.30 | 2026-01-02 10:15:00 | 1610.70 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-01-22 09:15:00 | 1701.40 | 2026-02-01 09:15:00 | 1732.00 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2026-01-22 12:15:00 | 1698.80 | 2026-02-01 09:15:00 | 1732.00 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2026-01-22 15:15:00 | 1695.00 | 2026-02-01 09:15:00 | 1732.00 | STOP_HIT | 1.00 | 2.18% |
| BUY | retest2 | 2026-01-23 11:00:00 | 1699.50 | 2026-02-01 09:15:00 | 1732.00 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2026-01-30 14:45:00 | 1747.00 | 2026-02-01 09:15:00 | 1732.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-02-03 13:00:00 | 1729.80 | 2026-02-04 09:15:00 | 1643.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-03 13:00:00 | 1729.80 | 2026-02-05 14:15:00 | 1645.10 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2026-02-17 13:15:00 | 1537.30 | 2026-02-20 09:15:00 | 1460.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:15:00 | 1537.30 | 2026-02-24 10:15:00 | 1383.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-18 09:15:00 | 1366.30 | 2026-03-19 14:15:00 | 1332.80 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1412.30 | 2026-03-27 13:15:00 | 1391.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-04-09 12:30:00 | 1456.10 | 2026-04-10 10:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-04-09 13:30:00 | 1455.40 | 2026-04-10 10:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-28 15:15:00 | 1412.50 | STOP_HIT | 1.00 | 0.31% |
