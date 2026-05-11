# HCLTECH (HCLTECH)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1198.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 218 |
| ALERT1 | 141 |
| ALERT2 | 138 |
| ALERT2_SKIP | 70 |
| ALERT3 | 358 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 161 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 167 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 177 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 125
- **Target hits / Stop hits / Partials:** 2 / 167 / 8
- **Avg / median % per leg:** 0.15% / -0.54%
- **Sum % (uncompounded):** 26.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 100 | 32 | 32.0% | 1 | 99 | 0 | -0.04% | -3.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.08% | 0.4% |
| BUY @ 3rd Alert (retest2) | 95 | 29 | 30.5% | 1 | 94 | 0 | -0.05% | -4.3% |
| SELL (all) | 77 | 20 | 26.0% | 1 | 68 | 8 | 0.40% | 30.8% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.51% | -1.5% |
| SELL @ 3rd Alert (retest2) | 74 | 19 | 25.7% | 1 | 65 | 8 | 0.44% | 32.3% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 8 | 0 | -0.14% | -1.1% |
| retest2 (combined) | 169 | 48 | 28.4% | 2 | 159 | 8 | 0.17% | 28.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 09:15:00 | 1080.75 | 1092.34 | 1093.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 1065.05 | 1086.88 | 1090.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 11:15:00 | 1079.00 | 1077.04 | 1082.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-18 12:00:00 | 1079.00 | 1077.04 | 1082.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 13:15:00 | 1082.95 | 1078.69 | 1081.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 13:45:00 | 1083.60 | 1078.69 | 1081.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 14:15:00 | 1079.80 | 1078.91 | 1081.73 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 09:15:00 | 1099.30 | 1083.28 | 1083.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 1103.20 | 1094.75 | 1090.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 12:15:00 | 1118.60 | 1119.15 | 1108.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 13:00:00 | 1118.60 | 1119.15 | 1108.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 1106.45 | 1115.98 | 1109.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 15:00:00 | 1106.45 | 1115.98 | 1109.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 1105.00 | 1113.79 | 1108.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:15:00 | 1104.35 | 1113.79 | 1108.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 1108.00 | 1112.50 | 1109.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 11:00:00 | 1108.00 | 1112.50 | 1109.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 11:15:00 | 1111.50 | 1112.30 | 1109.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 15:00:00 | 1114.10 | 1112.16 | 1109.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 12:30:00 | 1113.85 | 1111.46 | 1110.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 14:15:00 | 1113.80 | 1111.44 | 1110.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 14:45:00 | 1115.85 | 1112.51 | 1110.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 1126.15 | 1130.23 | 1126.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:30:00 | 1125.80 | 1130.23 | 1126.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 1127.80 | 1129.75 | 1126.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 09:15:00 | 1142.45 | 1129.75 | 1126.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 13:15:00 | 1137.60 | 1139.12 | 1139.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 13:15:00 | 1137.60 | 1139.12 | 1139.21 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 1142.50 | 1139.37 | 1139.26 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 09:15:00 | 1120.50 | 1136.25 | 1138.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 10:15:00 | 1114.30 | 1131.86 | 1136.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 1128.65 | 1126.29 | 1131.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-06 15:00:00 | 1128.65 | 1126.29 | 1131.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 1127.50 | 1126.54 | 1131.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-07 09:15:00 | 1121.05 | 1126.54 | 1131.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 1126.55 | 1126.54 | 1130.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-09 14:15:00 | 1115.35 | 1122.95 | 1125.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 10:15:00 | 1135.50 | 1123.20 | 1124.40 | SL hit (close>static) qty=1.00 sl=1133.80 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 1137.10 | 1125.98 | 1125.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 13:15:00 | 1138.35 | 1129.94 | 1127.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 10:15:00 | 1133.80 | 1133.83 | 1130.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-13 11:00:00 | 1133.80 | 1133.83 | 1130.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 1130.70 | 1133.74 | 1131.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 14:30:00 | 1131.15 | 1133.74 | 1131.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 15:15:00 | 1131.20 | 1133.23 | 1131.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 09:15:00 | 1132.50 | 1133.23 | 1131.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 09:15:00 | 1129.00 | 1132.38 | 1131.33 | SL hit (close<static) qty=1.00 sl=1130.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 1136.90 | 1139.24 | 1139.40 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 10:15:00 | 1148.45 | 1141.08 | 1140.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 11:15:00 | 1159.70 | 1144.81 | 1142.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 1160.40 | 1165.40 | 1159.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 1160.40 | 1165.40 | 1159.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1160.40 | 1165.40 | 1159.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 1160.40 | 1165.40 | 1159.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 1153.50 | 1161.68 | 1160.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:30:00 | 1151.30 | 1161.68 | 1160.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 1158.45 | 1161.03 | 1160.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 11:15:00 | 1161.00 | 1161.03 | 1160.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-06 14:15:00 | 1180.30 | 1186.06 | 1186.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 1180.30 | 1186.06 | 1186.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 1172.70 | 1182.24 | 1184.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 10:15:00 | 1118.25 | 1113.45 | 1121.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 10:15:00 | 1118.25 | 1113.45 | 1121.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 1118.25 | 1113.45 | 1121.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 11:00:00 | 1118.25 | 1113.45 | 1121.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 1119.90 | 1114.74 | 1121.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 12:00:00 | 1119.90 | 1114.74 | 1121.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 1137.50 | 1117.63 | 1119.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:00:00 | 1137.50 | 1117.63 | 1119.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 1133.80 | 1120.86 | 1121.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:15:00 | 1135.30 | 1120.86 | 1121.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 11:15:00 | 1138.60 | 1124.41 | 1122.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 1141.60 | 1127.85 | 1124.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 13:15:00 | 1153.10 | 1153.38 | 1142.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-17 13:45:00 | 1154.05 | 1153.38 | 1142.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 1149.00 | 1162.14 | 1158.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:00:00 | 1149.00 | 1162.14 | 1158.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 1150.50 | 1159.81 | 1158.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:15:00 | 1151.10 | 1159.81 | 1158.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 13:15:00 | 1153.15 | 1156.54 | 1156.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 09:15:00 | 1134.30 | 1151.89 | 1154.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 1115.75 | 1113.10 | 1119.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 1115.75 | 1113.10 | 1119.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 1115.75 | 1113.10 | 1119.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-26 09:30:00 | 1119.55 | 1113.10 | 1119.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1117.35 | 1115.09 | 1117.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 09:45:00 | 1118.25 | 1115.09 | 1117.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 1110.70 | 1114.21 | 1116.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:15:00 | 1109.80 | 1114.21 | 1116.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 12:15:00 | 1118.65 | 1115.39 | 1116.88 | SL hit (close>static) qty=1.00 sl=1117.60 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 15:15:00 | 1115.65 | 1111.60 | 1111.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 09:15:00 | 1127.15 | 1114.71 | 1112.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 1128.95 | 1129.07 | 1122.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-02 11:00:00 | 1128.95 | 1129.07 | 1122.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 1122.90 | 1128.91 | 1124.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 1122.90 | 1128.91 | 1124.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 1132.10 | 1129.55 | 1124.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:30:00 | 1133.60 | 1128.44 | 1125.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 10:15:00 | 1121.25 | 1127.00 | 1124.85 | SL hit (close<static) qty=1.00 sl=1121.75 alert=retest2 |

### Cycle 13 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 1128.85 | 1140.60 | 1141.50 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 1171.60 | 1143.63 | 1141.25 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 1171.45 | 1180.13 | 1180.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 1164.90 | 1175.94 | 1178.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 10:15:00 | 1152.15 | 1151.61 | 1157.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 10:30:00 | 1151.00 | 1151.61 | 1157.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 1162.45 | 1154.61 | 1158.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 13:00:00 | 1162.45 | 1154.61 | 1158.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 1161.90 | 1156.07 | 1158.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:30:00 | 1157.50 | 1156.27 | 1158.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 09:15:00 | 1167.15 | 1158.88 | 1159.16 | SL hit (close>static) qty=1.00 sl=1163.20 alert=retest2 |

### Cycle 16 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 1167.25 | 1160.55 | 1159.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 1171.75 | 1162.79 | 1160.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 12:15:00 | 1174.65 | 1174.92 | 1169.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 13:00:00 | 1174.65 | 1174.92 | 1169.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 1175.00 | 1175.06 | 1170.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 10:00:00 | 1177.95 | 1175.64 | 1171.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-15 09:15:00 | 1295.75 | 1286.34 | 1280.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 1266.60 | 1290.83 | 1293.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 13:15:00 | 1258.10 | 1266.29 | 1273.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 15:15:00 | 1266.05 | 1265.85 | 1272.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 09:15:00 | 1258.95 | 1265.85 | 1272.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1263.05 | 1260.60 | 1264.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:45:00 | 1263.90 | 1260.60 | 1264.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 1254.55 | 1259.53 | 1262.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-28 11:15:00 | 1249.00 | 1258.15 | 1261.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 12:45:00 | 1249.50 | 1244.21 | 1246.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-05 11:15:00 | 1242.15 | 1241.29 | 1241.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 1251.85 | 1241.20 | 1240.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 1251.85 | 1241.20 | 1240.28 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 15:15:00 | 1235.70 | 1239.93 | 1240.05 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 09:15:00 | 1255.25 | 1243.00 | 1241.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 10:15:00 | 1260.80 | 1246.56 | 1243.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 14:15:00 | 1248.60 | 1250.40 | 1246.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 15:00:00 | 1248.60 | 1250.40 | 1246.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 1249.90 | 1250.30 | 1246.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 1251.95 | 1250.30 | 1246.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 14:15:00 | 1244.60 | 1257.64 | 1255.77 | SL hit (close<static) qty=1.00 sl=1246.75 alert=retest2 |

### Cycle 21 — SELL (started 2023-10-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 09:15:00 | 1238.00 | 1251.52 | 1253.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 11:15:00 | 1232.80 | 1246.49 | 1250.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 09:15:00 | 1254.60 | 1236.90 | 1242.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1254.60 | 1236.90 | 1242.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1254.60 | 1236.90 | 1242.84 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 14:15:00 | 1255.10 | 1245.86 | 1245.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 09:15:00 | 1274.70 | 1253.41 | 1249.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 12:15:00 | 1275.00 | 1275.28 | 1270.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 12:30:00 | 1273.00 | 1275.28 | 1270.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1268.50 | 1274.03 | 1270.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 1268.50 | 1274.03 | 1270.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 1271.35 | 1273.49 | 1270.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:15:00 | 1268.65 | 1273.49 | 1270.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 1273.30 | 1273.46 | 1270.78 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 1266.00 | 1269.74 | 1269.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 1250.15 | 1265.82 | 1268.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 12:15:00 | 1234.65 | 1233.66 | 1240.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-26 13:00:00 | 1234.65 | 1233.66 | 1240.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1247.10 | 1234.71 | 1238.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 1247.10 | 1234.71 | 1238.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 1251.90 | 1238.15 | 1240.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 1251.90 | 1238.15 | 1240.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 1261.30 | 1242.78 | 1242.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 13:15:00 | 1265.25 | 1250.78 | 1246.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 10:15:00 | 1262.05 | 1262.39 | 1257.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 10:15:00 | 1262.05 | 1262.39 | 1257.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 10:15:00 | 1262.05 | 1262.39 | 1257.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-31 10:45:00 | 1261.90 | 1262.39 | 1257.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 1266.10 | 1271.78 | 1267.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 1266.10 | 1271.78 | 1267.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 1267.90 | 1271.00 | 1267.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 1272.35 | 1266.05 | 1265.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:30:00 | 1271.25 | 1268.12 | 1266.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 11:45:00 | 1271.40 | 1268.49 | 1266.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 1274.55 | 1270.18 | 1269.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 1270.80 | 1271.41 | 1270.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:00:00 | 1270.80 | 1271.41 | 1270.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 1269.90 | 1271.11 | 1270.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 11:45:00 | 1269.85 | 1271.11 | 1270.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 1268.85 | 1270.66 | 1270.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:30:00 | 1268.55 | 1270.66 | 1270.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-06 13:15:00 | 1266.25 | 1269.78 | 1269.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 13:15:00 | 1266.25 | 1269.78 | 1269.97 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 09:15:00 | 1276.35 | 1271.06 | 1270.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 09:15:00 | 1277.65 | 1274.17 | 1272.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 1271.55 | 1273.65 | 1272.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 10:15:00 | 1271.55 | 1273.65 | 1272.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1271.55 | 1273.65 | 1272.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 1271.55 | 1273.65 | 1272.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1271.75 | 1273.27 | 1272.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 13:15:00 | 1276.65 | 1273.75 | 1272.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-08 14:15:00 | 1267.40 | 1272.20 | 1272.07 | SL hit (close<static) qty=1.00 sl=1271.10 alert=retest2 |

### Cycle 27 — SELL (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 15:15:00 | 1267.50 | 1271.26 | 1271.66 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 09:15:00 | 1275.60 | 1272.13 | 1272.01 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 10:15:00 | 1270.40 | 1271.78 | 1271.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 13:15:00 | 1269.00 | 1271.30 | 1271.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 1260.10 | 1258.78 | 1263.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-12 18:15:00 | 1260.10 | 1258.78 | 1263.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1260.10 | 1258.78 | 1263.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 11:00:00 | 1256.00 | 1258.74 | 1262.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 1281.65 | 1263.50 | 1262.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1281.65 | 1263.50 | 1262.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 1295.75 | 1277.24 | 1270.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 14:15:00 | 1309.35 | 1310.69 | 1298.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 15:00:00 | 1309.35 | 1310.69 | 1298.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 1323.25 | 1326.73 | 1322.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:00:00 | 1323.25 | 1326.73 | 1322.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 13:15:00 | 1321.60 | 1325.71 | 1321.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 13:30:00 | 1321.65 | 1325.71 | 1321.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 14:15:00 | 1329.60 | 1326.49 | 1322.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 1332.60 | 1327.19 | 1323.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 12:45:00 | 1333.80 | 1328.82 | 1325.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 09:15:00 | 1318.00 | 1327.00 | 1325.73 | SL hit (close<static) qty=1.00 sl=1321.40 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 1311.60 | 1323.92 | 1324.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 1308.85 | 1319.19 | 1322.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 14:15:00 | 1316.90 | 1308.00 | 1312.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 1316.90 | 1308.00 | 1312.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1316.90 | 1308.00 | 1312.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 15:00:00 | 1316.90 | 1308.00 | 1312.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1319.90 | 1310.38 | 1313.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:15:00 | 1329.95 | 1310.38 | 1313.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2023-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 10:15:00 | 1336.50 | 1319.31 | 1317.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 1341.30 | 1334.13 | 1328.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 10:15:00 | 1334.10 | 1335.42 | 1330.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 11:00:00 | 1334.10 | 1335.42 | 1330.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 1331.00 | 1335.02 | 1331.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:00:00 | 1331.00 | 1335.02 | 1331.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 1336.70 | 1335.36 | 1332.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:30:00 | 1332.85 | 1335.36 | 1332.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 1342.00 | 1336.83 | 1333.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 14:30:00 | 1343.00 | 1338.18 | 1335.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 09:15:00 | 1318.50 | 1333.73 | 1333.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 1318.50 | 1333.73 | 1333.86 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 1333.55 | 1327.24 | 1327.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 1351.85 | 1334.14 | 1330.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 14:15:00 | 1374.60 | 1376.50 | 1367.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-12 14:45:00 | 1371.00 | 1376.50 | 1367.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1369.70 | 1374.42 | 1368.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 10:00:00 | 1369.70 | 1374.42 | 1368.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 1368.15 | 1373.16 | 1368.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 09:15:00 | 1412.20 | 1367.78 | 1366.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 1441.95 | 1469.77 | 1472.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1441.95 | 1469.77 | 1472.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1437.25 | 1463.27 | 1469.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1441.00 | 1437.32 | 1449.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 10:00:00 | 1441.00 | 1437.32 | 1449.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 11:15:00 | 1444.65 | 1439.48 | 1448.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:30:00 | 1450.40 | 1439.48 | 1448.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 1461.05 | 1443.92 | 1448.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 15:00:00 | 1461.05 | 1443.92 | 1448.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 15:15:00 | 1459.20 | 1446.98 | 1449.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 09:15:00 | 1442.75 | 1446.98 | 1449.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 1463.70 | 1451.48 | 1451.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 1465.15 | 1454.22 | 1452.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 1462.75 | 1463.09 | 1458.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 13:00:00 | 1462.75 | 1463.09 | 1458.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 13:15:00 | 1466.40 | 1463.75 | 1459.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 13:30:00 | 1461.65 | 1463.75 | 1459.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1472.80 | 1472.85 | 1468.55 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 09:15:00 | 1461.90 | 1466.13 | 1466.47 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 1470.00 | 1466.90 | 1466.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 11:15:00 | 1473.40 | 1468.20 | 1467.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 1472.90 | 1475.82 | 1472.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 09:15:00 | 1472.90 | 1475.82 | 1472.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 09:15:00 | 1472.90 | 1475.82 | 1472.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:00:00 | 1472.90 | 1475.82 | 1472.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1465.60 | 1473.78 | 1471.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 1466.00 | 1473.78 | 1471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1466.00 | 1472.22 | 1471.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:45:00 | 1462.25 | 1472.22 | 1471.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 1468.90 | 1471.05 | 1470.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 15:00:00 | 1468.90 | 1471.05 | 1470.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 1470.00 | 1470.84 | 1470.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:15:00 | 1449.30 | 1470.84 | 1470.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 09:15:00 | 1439.60 | 1464.59 | 1467.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 14:15:00 | 1437.60 | 1449.82 | 1458.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-05 09:15:00 | 1430.60 | 1427.63 | 1438.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-05 09:45:00 | 1427.70 | 1427.63 | 1438.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 1436.15 | 1430.62 | 1436.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:00:00 | 1436.15 | 1430.62 | 1436.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 14:15:00 | 1432.60 | 1431.01 | 1436.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 14:45:00 | 1433.55 | 1431.01 | 1436.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 15:15:00 | 1434.95 | 1431.80 | 1436.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 09:15:00 | 1437.95 | 1431.80 | 1436.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1445.25 | 1434.49 | 1437.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 10:15:00 | 1447.45 | 1434.49 | 1437.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 10:15:00 | 1438.35 | 1435.26 | 1437.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 10:30:00 | 1444.30 | 1435.26 | 1437.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 11:15:00 | 1444.40 | 1437.09 | 1437.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-08 11:45:00 | 1441.65 | 1437.09 | 1437.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 12:15:00 | 1445.00 | 1438.67 | 1438.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 09:15:00 | 1475.60 | 1448.29 | 1443.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 14:15:00 | 1458.90 | 1459.94 | 1451.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 15:00:00 | 1458.90 | 1459.94 | 1451.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 1475.75 | 1483.66 | 1476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 1475.75 | 1483.66 | 1476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 1486.10 | 1484.15 | 1477.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 09:15:00 | 1509.35 | 1484.52 | 1478.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 11:15:00 | 1561.20 | 1564.42 | 1564.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 1561.20 | 1564.42 | 1564.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 12:15:00 | 1555.00 | 1562.54 | 1563.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 1557.80 | 1554.52 | 1558.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 1557.80 | 1554.52 | 1558.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1557.80 | 1554.52 | 1558.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:45:00 | 1560.00 | 1554.52 | 1558.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1549.60 | 1553.54 | 1557.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 12:00:00 | 1541.95 | 1551.22 | 1556.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 13:00:00 | 1542.05 | 1549.39 | 1555.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 10:00:00 | 1544.30 | 1539.27 | 1547.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 10:30:00 | 1540.05 | 1540.95 | 1547.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 1543.90 | 1541.54 | 1547.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:15:00 | 1541.95 | 1541.54 | 1547.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1546.90 | 1542.61 | 1547.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:00:00 | 1546.90 | 1542.61 | 1547.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1560.20 | 1546.13 | 1548.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 1560.20 | 1546.13 | 1548.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-24 14:15:00 | 1578.80 | 1552.67 | 1551.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 1578.80 | 1552.67 | 1551.25 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 12:15:00 | 1542.10 | 1551.78 | 1552.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 13:15:00 | 1537.05 | 1548.84 | 1550.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 14:15:00 | 1551.40 | 1549.35 | 1550.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 14:15:00 | 1551.40 | 1549.35 | 1550.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 1551.40 | 1549.35 | 1550.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 1551.40 | 1549.35 | 1550.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1549.50 | 1549.38 | 1550.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 1571.00 | 1549.38 | 1550.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 1564.25 | 1552.35 | 1551.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 11:15:00 | 1586.60 | 1562.04 | 1556.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 1558.30 | 1573.98 | 1569.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 14:15:00 | 1558.30 | 1573.98 | 1569.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1558.30 | 1573.98 | 1569.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 1558.30 | 1573.98 | 1569.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 1559.80 | 1571.14 | 1568.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 1561.60 | 1571.14 | 1568.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 1564.30 | 1568.26 | 1567.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:45:00 | 1559.90 | 1568.26 | 1567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 1573.30 | 1571.51 | 1569.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:45:00 | 1570.45 | 1571.51 | 1569.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 1570.80 | 1571.94 | 1569.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:30:00 | 1578.95 | 1576.58 | 1572.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 13:30:00 | 1577.00 | 1576.81 | 1573.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 14:30:00 | 1576.65 | 1576.85 | 1574.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 15:00:00 | 1577.00 | 1576.85 | 1574.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 1578.00 | 1577.08 | 1574.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 1598.50 | 1577.08 | 1574.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 14:00:00 | 1581.65 | 1584.43 | 1580.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 14:30:00 | 1582.80 | 1584.25 | 1580.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 15:00:00 | 1583.50 | 1584.25 | 1580.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 1582.80 | 1583.91 | 1580.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:30:00 | 1584.30 | 1583.91 | 1580.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 1577.50 | 1582.63 | 1580.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-05 12:15:00 | 1570.80 | 1579.29 | 1579.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 12:15:00 | 1570.80 | 1579.29 | 1579.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 13:15:00 | 1567.70 | 1576.97 | 1578.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 1590.10 | 1573.29 | 1575.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 09:15:00 | 1590.10 | 1573.29 | 1575.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1590.10 | 1573.29 | 1575.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:30:00 | 1598.10 | 1573.29 | 1575.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 1593.00 | 1577.23 | 1577.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 10:30:00 | 1590.90 | 1577.23 | 1577.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 11:15:00 | 1596.00 | 1580.99 | 1578.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 12:15:00 | 1613.05 | 1587.40 | 1582.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 1619.60 | 1627.14 | 1618.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 1619.60 | 1627.14 | 1618.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1619.60 | 1627.14 | 1618.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 1619.60 | 1627.14 | 1618.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 1623.75 | 1626.46 | 1618.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 13:30:00 | 1632.25 | 1626.38 | 1620.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:00:00 | 1635.50 | 1628.21 | 1621.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 09:15:00 | 1655.25 | 1627.55 | 1622.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 10:15:00 | 1646.30 | 1665.27 | 1665.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 10:15:00 | 1646.30 | 1665.27 | 1665.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 10:15:00 | 1644.80 | 1655.91 | 1660.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 09:15:00 | 1654.80 | 1647.29 | 1653.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 09:15:00 | 1654.80 | 1647.29 | 1653.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1654.80 | 1647.29 | 1653.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:45:00 | 1655.50 | 1647.29 | 1653.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1663.40 | 1650.51 | 1654.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 1663.40 | 1650.51 | 1654.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1663.00 | 1653.01 | 1654.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:30:00 | 1664.10 | 1653.01 | 1654.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-02-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 13:15:00 | 1671.15 | 1658.08 | 1656.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 14:15:00 | 1687.50 | 1663.96 | 1659.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 12:15:00 | 1675.40 | 1676.63 | 1668.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-23 13:00:00 | 1675.40 | 1676.63 | 1668.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 13:15:00 | 1670.50 | 1675.41 | 1668.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 13:45:00 | 1669.25 | 1675.41 | 1668.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 1665.90 | 1673.50 | 1668.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 15:00:00 | 1665.90 | 1673.50 | 1668.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1664.40 | 1671.68 | 1668.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:15:00 | 1654.05 | 1671.68 | 1668.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1653.65 | 1668.08 | 1666.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:00:00 | 1653.65 | 1668.08 | 1666.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 1646.80 | 1663.82 | 1665.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 1640.75 | 1659.21 | 1662.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 09:15:00 | 1655.35 | 1653.84 | 1658.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 09:15:00 | 1655.35 | 1653.84 | 1658.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1655.35 | 1653.84 | 1658.37 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 1663.60 | 1660.74 | 1660.64 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1658.50 | 1660.29 | 1660.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1647.20 | 1656.60 | 1658.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 1652.65 | 1651.28 | 1654.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 1652.65 | 1651.28 | 1654.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1660.10 | 1653.04 | 1655.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:30:00 | 1659.35 | 1653.04 | 1655.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1661.30 | 1654.69 | 1655.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:45:00 | 1663.70 | 1654.69 | 1655.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-02-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 13:15:00 | 1665.40 | 1656.84 | 1656.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 14:15:00 | 1669.75 | 1659.42 | 1658.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 10:15:00 | 1656.30 | 1661.39 | 1659.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 10:15:00 | 1656.30 | 1661.39 | 1659.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 1656.30 | 1661.39 | 1659.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:45:00 | 1652.85 | 1661.39 | 1659.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 1657.35 | 1660.58 | 1659.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:00:00 | 1657.35 | 1660.58 | 1659.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 12:15:00 | 1650.00 | 1658.46 | 1658.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 13:15:00 | 1645.90 | 1655.95 | 1657.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 12:15:00 | 1627.65 | 1621.51 | 1629.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 13:00:00 | 1627.65 | 1621.51 | 1629.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 1633.40 | 1623.89 | 1629.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 13:45:00 | 1634.30 | 1623.89 | 1629.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1644.60 | 1628.03 | 1630.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 1644.60 | 1628.03 | 1630.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 1648.00 | 1632.02 | 1632.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 1644.15 | 1632.02 | 1632.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 09:15:00 | 1648.80 | 1635.38 | 1633.86 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 1635.00 | 1639.30 | 1639.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 10:15:00 | 1629.50 | 1637.15 | 1638.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 1635.85 | 1629.92 | 1633.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 14:15:00 | 1635.85 | 1629.92 | 1633.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 14:15:00 | 1635.85 | 1629.92 | 1633.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-13 15:00:00 | 1635.85 | 1629.92 | 1633.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 15:15:00 | 1624.50 | 1628.84 | 1633.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:15:00 | 1638.50 | 1628.84 | 1633.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1646.55 | 1632.38 | 1634.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 1646.55 | 1632.38 | 1634.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 1645.85 | 1635.07 | 1635.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 11:00:00 | 1645.85 | 1635.07 | 1635.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 11:15:00 | 1656.70 | 1639.40 | 1637.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 12:15:00 | 1659.85 | 1643.49 | 1639.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 12:15:00 | 1657.45 | 1658.37 | 1651.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-15 13:00:00 | 1657.45 | 1658.37 | 1651.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 1653.00 | 1657.30 | 1651.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 13:45:00 | 1650.00 | 1657.30 | 1651.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1646.30 | 1655.10 | 1650.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 1646.30 | 1655.10 | 1650.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1648.60 | 1653.80 | 1650.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 1657.10 | 1653.80 | 1650.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 11:00:00 | 1651.05 | 1652.64 | 1650.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 14:15:00 | 1652.55 | 1652.59 | 1651.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 15:15:00 | 1642.00 | 1649.22 | 1649.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 15:15:00 | 1642.00 | 1649.22 | 1649.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 1620.00 | 1643.37 | 1647.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1616.50 | 1600.72 | 1611.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1616.50 | 1600.72 | 1611.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1616.50 | 1600.72 | 1611.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 11:45:00 | 1596.80 | 1599.39 | 1609.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 14:45:00 | 1594.30 | 1598.79 | 1606.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 1515.00 | 1599.03 | 1606.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 09:15:00 | 1516.96 | 1586.35 | 1599.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 09:15:00 | 1514.58 | 1586.35 | 1599.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 14:15:00 | 1563.35 | 1561.66 | 1572.04 | SL hit (close>ema200) qty=0.50 sl=1561.66 alert=retest2 |

### Cycle 58 — BUY (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 09:15:00 | 1546.30 | 1542.19 | 1541.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 1555.80 | 1545.14 | 1543.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 11:15:00 | 1546.05 | 1546.91 | 1544.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-09 11:45:00 | 1548.40 | 1546.91 | 1544.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 12:15:00 | 1540.05 | 1545.54 | 1544.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 12:30:00 | 1540.30 | 1545.54 | 1544.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 1540.65 | 1544.56 | 1544.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 1547.15 | 1543.84 | 1543.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-10 10:00:00 | 1543.90 | 1543.85 | 1543.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 1537.85 | 1543.96 | 1544.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 1537.85 | 1543.96 | 1544.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 1531.65 | 1540.11 | 1542.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 1468.40 | 1455.35 | 1465.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 1468.40 | 1455.35 | 1465.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1468.40 | 1455.35 | 1465.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:45:00 | 1466.50 | 1455.35 | 1465.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 1470.30 | 1458.34 | 1466.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:30:00 | 1457.75 | 1459.84 | 1465.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 09:15:00 | 1484.90 | 1467.81 | 1467.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 1484.90 | 1467.81 | 1467.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 10:15:00 | 1493.50 | 1472.94 | 1470.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 11:15:00 | 1486.05 | 1486.15 | 1480.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 11:45:00 | 1484.75 | 1486.15 | 1480.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 1483.20 | 1485.11 | 1481.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:00:00 | 1483.20 | 1485.11 | 1481.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 1478.80 | 1483.85 | 1480.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 1478.80 | 1483.85 | 1480.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 1482.00 | 1483.48 | 1481.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:15:00 | 1492.65 | 1483.48 | 1481.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 14:15:00 | 1471.05 | 1500.31 | 1497.23 | SL hit (close<static) qty=1.00 sl=1478.70 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 1385.00 | 1473.52 | 1485.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 10:15:00 | 1375.10 | 1399.58 | 1432.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 1354.40 | 1351.30 | 1365.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1334.60 | 1356.75 | 1362.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1318.10 | 1318.72 | 1330.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 1327.20 | 1318.72 | 1330.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1330.00 | 1320.98 | 1330.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 1330.00 | 1320.98 | 1330.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 1334.30 | 1323.64 | 1330.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-09 11:15:00 | 1334.30 | 1323.64 | 1330.62 | SL hit (close>ema400) qty=1.00 sl=1330.62 alert=retest1 |

### Cycle 62 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1321.75 | 1318.71 | 1318.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 1331.85 | 1321.34 | 1319.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 1332.35 | 1339.13 | 1332.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 12:15:00 | 1332.35 | 1339.13 | 1332.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1332.35 | 1339.13 | 1332.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:00:00 | 1332.35 | 1339.13 | 1332.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1332.85 | 1337.87 | 1332.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:30:00 | 1332.60 | 1337.87 | 1332.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1347.10 | 1339.72 | 1333.44 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 1331.15 | 1335.29 | 1335.69 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 13:15:00 | 1346.75 | 1337.01 | 1336.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 1355.95 | 1346.47 | 1342.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 1349.00 | 1350.18 | 1346.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1349.00 | 1350.18 | 1346.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1349.00 | 1350.18 | 1346.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:30:00 | 1362.65 | 1354.13 | 1351.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 1360.10 | 1356.97 | 1354.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1336.45 | 1351.16 | 1352.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1336.45 | 1351.16 | 1352.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 1332.35 | 1347.40 | 1350.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 1336.95 | 1336.60 | 1342.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:30:00 | 1337.25 | 1336.60 | 1342.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1320.85 | 1332.27 | 1338.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 1337.25 | 1332.27 | 1338.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1326.70 | 1329.83 | 1336.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:00:00 | 1320.35 | 1326.94 | 1333.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1254.33 | 1310.24 | 1320.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 1312.45 | 1309.04 | 1318.14 | SL hit (close>ema200) qty=0.50 sl=1309.04 alert=retest2 |

### Cycle 66 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1352.40 | 1323.90 | 1323.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1373.35 | 1345.61 | 1335.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1399.60 | 1413.87 | 1393.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 1399.60 | 1413.87 | 1393.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1430.50 | 1440.66 | 1435.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 1430.00 | 1440.66 | 1435.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1429.60 | 1438.44 | 1435.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:30:00 | 1428.00 | 1438.44 | 1435.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1431.05 | 1435.55 | 1434.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 1431.05 | 1435.55 | 1434.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 1431.20 | 1434.68 | 1434.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 1433.65 | 1434.68 | 1434.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 09:15:00 | 1428.55 | 1433.45 | 1433.95 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 13:15:00 | 1437.95 | 1434.12 | 1434.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 11:15:00 | 1446.95 | 1437.84 | 1435.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 1432.50 | 1439.88 | 1438.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1432.50 | 1439.88 | 1438.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1432.50 | 1439.88 | 1438.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 10:00:00 | 1432.50 | 1439.88 | 1438.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1437.95 | 1439.50 | 1438.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:30:00 | 1441.85 | 1440.06 | 1438.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 1441.70 | 1440.55 | 1439.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:00:00 | 1440.90 | 1444.27 | 1444.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 1440.20 | 1443.46 | 1443.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 1440.20 | 1443.46 | 1443.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 1430.00 | 1439.21 | 1441.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 10:15:00 | 1442.30 | 1439.83 | 1441.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 10:15:00 | 1442.30 | 1439.83 | 1441.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1442.30 | 1439.83 | 1441.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 1442.30 | 1439.83 | 1441.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1442.15 | 1440.29 | 1441.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 1440.10 | 1440.29 | 1441.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:15:00 | 1439.75 | 1440.66 | 1441.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 14:15:00 | 1449.60 | 1442.49 | 1442.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 14:15:00 | 1449.60 | 1442.49 | 1442.33 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 1431.80 | 1440.58 | 1441.66 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 13:15:00 | 1450.35 | 1443.40 | 1442.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 1461.00 | 1453.36 | 1448.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 1467.00 | 1467.22 | 1459.85 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:15:00 | 1480.95 | 1467.22 | 1459.85 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1521.70 | 1518.99 | 1508.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 1508.85 | 1518.99 | 1508.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1510.35 | 1515.88 | 1509.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:30:00 | 1513.35 | 1515.88 | 1509.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1511.50 | 1515.00 | 1509.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 1518.95 | 1516.28 | 1510.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1509.15 | 1527.08 | 1523.45 | SL hit (close<ema400) qty=1.00 sl=1523.45 alert=retest1 |

### Cycle 73 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1492.70 | 1520.20 | 1520.66 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1521.25 | 1517.84 | 1517.65 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 1511.70 | 1516.61 | 1517.11 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1526.70 | 1518.65 | 1517.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 1557.90 | 1526.50 | 1521.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 1568.90 | 1571.53 | 1555.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:30:00 | 1571.70 | 1571.53 | 1555.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1571.65 | 1570.99 | 1558.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 1585.50 | 1574.38 | 1567.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:45:00 | 1588.30 | 1593.54 | 1586.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 1576.05 | 1583.38 | 1583.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 14:15:00 | 1576.05 | 1583.38 | 1583.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 1562.00 | 1578.55 | 1581.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 1579.80 | 1578.43 | 1580.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 1579.80 | 1578.43 | 1580.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1579.80 | 1578.43 | 1580.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:45:00 | 1577.10 | 1578.43 | 1580.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1571.25 | 1576.99 | 1579.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1573.65 | 1576.99 | 1579.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1589.85 | 1579.57 | 1580.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1593.15 | 1579.57 | 1580.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 1597.35 | 1583.12 | 1582.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 1619.85 | 1594.23 | 1589.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1624.50 | 1626.53 | 1618.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 1624.60 | 1625.62 | 1619.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 1624.60 | 1625.62 | 1619.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:45:00 | 1622.00 | 1625.62 | 1619.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1625.20 | 1624.94 | 1619.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:15:00 | 1627.95 | 1624.94 | 1619.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 1627.20 | 1628.02 | 1622.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1618.30 | 1637.38 | 1635.46 | SL hit (close<static) qty=1.00 sl=1619.20 alert=retest2 |

### Cycle 79 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1620.45 | 1633.99 | 1634.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 13:15:00 | 1613.75 | 1625.82 | 1629.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1597.45 | 1578.38 | 1595.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1597.45 | 1578.38 | 1595.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1597.45 | 1578.38 | 1595.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 1600.95 | 1578.38 | 1595.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1594.40 | 1581.59 | 1595.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:45:00 | 1604.90 | 1581.59 | 1595.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 1600.10 | 1585.29 | 1595.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 12:00:00 | 1600.10 | 1585.29 | 1595.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 1587.45 | 1585.72 | 1594.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1582.50 | 1585.08 | 1593.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 1583.00 | 1584.20 | 1592.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 1580.70 | 1584.20 | 1592.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 1598.00 | 1594.86 | 1594.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1598.00 | 1594.86 | 1594.75 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 1589.05 | 1593.70 | 1594.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 1568.05 | 1588.36 | 1591.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 1588.95 | 1577.86 | 1584.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 1588.95 | 1577.86 | 1584.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1588.95 | 1577.86 | 1584.50 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 13:15:00 | 1591.50 | 1587.91 | 1587.88 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1579.05 | 1586.66 | 1587.38 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 10:15:00 | 1593.20 | 1587.97 | 1587.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 11:15:00 | 1594.85 | 1589.35 | 1588.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 14:15:00 | 1584.45 | 1590.52 | 1589.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 14:15:00 | 1584.45 | 1590.52 | 1589.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1584.45 | 1590.52 | 1589.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 1584.45 | 1590.52 | 1589.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 1583.25 | 1589.06 | 1588.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 1585.65 | 1589.06 | 1588.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1602.50 | 1591.75 | 1590.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1607.25 | 1594.85 | 1591.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1614.55 | 1592.62 | 1591.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 1669.10 | 1674.39 | 1674.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 1669.10 | 1674.39 | 1674.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 1665.00 | 1671.39 | 1673.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1681.00 | 1670.59 | 1671.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 1681.00 | 1670.59 | 1671.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1681.00 | 1670.59 | 1671.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 1680.95 | 1670.59 | 1671.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1677.95 | 1672.06 | 1672.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:15:00 | 1684.00 | 1672.06 | 1672.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 11:15:00 | 1686.90 | 1675.03 | 1673.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 12:15:00 | 1700.70 | 1680.17 | 1676.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 15:15:00 | 1706.00 | 1710.89 | 1700.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 09:15:00 | 1703.55 | 1710.89 | 1700.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1708.95 | 1710.50 | 1701.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 1702.85 | 1710.50 | 1701.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1738.30 | 1723.37 | 1713.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 1747.15 | 1723.37 | 1713.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:00:00 | 1743.00 | 1730.89 | 1718.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 14:45:00 | 1747.30 | 1739.31 | 1726.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 1773.00 | 1782.12 | 1782.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 10:15:00 | 1773.00 | 1782.12 | 1782.49 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 15:15:00 | 1794.60 | 1782.83 | 1782.15 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1754.90 | 1777.25 | 1779.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 1744.60 | 1764.02 | 1772.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1762.45 | 1758.10 | 1765.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 1762.45 | 1758.10 | 1765.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1762.45 | 1758.10 | 1765.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:45:00 | 1763.55 | 1758.10 | 1765.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1755.55 | 1753.71 | 1759.82 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 1781.90 | 1765.33 | 1763.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 1790.60 | 1777.47 | 1770.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1774.75 | 1779.73 | 1773.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 1774.75 | 1779.73 | 1773.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1783.20 | 1780.43 | 1774.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1795.00 | 1780.43 | 1774.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 1793.50 | 1784.44 | 1779.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 1772.05 | 1807.49 | 1808.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 09:15:00 | 1772.05 | 1807.49 | 1808.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 10:15:00 | 1766.90 | 1799.37 | 1804.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1748.00 | 1744.94 | 1759.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 10:30:00 | 1750.00 | 1744.94 | 1759.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 1761.50 | 1748.86 | 1759.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 1756.80 | 1748.86 | 1759.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1741.00 | 1747.29 | 1757.47 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 1767.00 | 1755.02 | 1754.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 1773.95 | 1758.81 | 1756.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1756.65 | 1763.69 | 1760.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 1756.65 | 1763.69 | 1760.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1756.65 | 1763.69 | 1760.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 1756.65 | 1763.69 | 1760.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1761.85 | 1763.32 | 1760.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 1764.50 | 1763.32 | 1760.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 1780.55 | 1798.78 | 1800.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1780.55 | 1798.78 | 1800.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 1774.55 | 1786.56 | 1791.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 1779.40 | 1778.62 | 1784.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 1779.40 | 1778.62 | 1784.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 1779.35 | 1773.62 | 1778.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 1779.35 | 1773.62 | 1778.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1789.70 | 1776.83 | 1779.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1789.70 | 1776.83 | 1779.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1784.00 | 1778.27 | 1779.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1805.25 | 1778.27 | 1779.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1812.45 | 1785.10 | 1782.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1830.60 | 1810.07 | 1798.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 1809.90 | 1815.28 | 1805.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 13:15:00 | 1809.90 | 1815.28 | 1805.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 1809.90 | 1815.28 | 1805.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 1800.25 | 1815.28 | 1805.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1810.35 | 1814.29 | 1805.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:00:00 | 1810.35 | 1814.29 | 1805.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1828.60 | 1816.79 | 1808.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 12:30:00 | 1854.20 | 1840.35 | 1828.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:30:00 | 1854.00 | 1843.79 | 1831.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 1853.55 | 1859.27 | 1859.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1848.00 | 1857.02 | 1858.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1848.00 | 1857.02 | 1858.22 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 1862.45 | 1858.65 | 1858.42 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 1854.55 | 1857.83 | 1858.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 12:15:00 | 1852.90 | 1856.85 | 1857.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1837.65 | 1834.42 | 1842.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 10:00:00 | 1837.65 | 1834.42 | 1842.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1852.80 | 1838.10 | 1843.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1852.80 | 1838.10 | 1843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1855.05 | 1841.49 | 1844.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1860.80 | 1841.49 | 1844.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 1845.00 | 1843.90 | 1844.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 1859.65 | 1843.90 | 1844.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1843.80 | 1843.88 | 1844.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 10:30:00 | 1838.90 | 1844.70 | 1844.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 1852.00 | 1846.16 | 1845.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 11:15:00 | 1852.00 | 1846.16 | 1845.56 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1842.80 | 1845.27 | 1845.32 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 09:15:00 | 1857.00 | 1847.61 | 1846.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 12:15:00 | 1874.85 | 1858.88 | 1853.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1862.40 | 1865.62 | 1858.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 1862.40 | 1865.62 | 1858.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1862.40 | 1865.62 | 1858.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:00:00 | 1862.40 | 1865.62 | 1858.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 1861.60 | 1864.81 | 1859.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:30:00 | 1856.05 | 1864.81 | 1859.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 1856.55 | 1863.16 | 1858.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 1855.95 | 1863.16 | 1858.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 1854.45 | 1861.42 | 1858.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:30:00 | 1854.05 | 1861.42 | 1858.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1859.00 | 1864.07 | 1860.88 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 1847.10 | 1857.33 | 1858.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 1836.75 | 1853.21 | 1856.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 1853.75 | 1853.32 | 1856.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:15:00 | 1816.25 | 1853.32 | 1856.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1767.00 | 1764.85 | 1782.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 1821.85 | 1780.20 | 1781.76 | SL hit (close>ema400) qty=1.00 sl=1781.76 alert=retest1 |

### Cycle 102 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1833.60 | 1790.88 | 1786.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1842.40 | 1807.62 | 1795.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1823.05 | 1823.82 | 1808.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 1823.05 | 1823.82 | 1808.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1863.00 | 1870.81 | 1861.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 1867.50 | 1870.81 | 1861.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1869.55 | 1870.56 | 1862.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-14 09:15:00 | 1895.90 | 1867.48 | 1863.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 10:15:00 | 1855.90 | 1865.57 | 1863.58 | SL hit (close<static) qty=1.00 sl=1859.10 alert=retest2 |

### Cycle 103 — SELL (started 2024-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 12:15:00 | 1853.45 | 1861.62 | 1862.03 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 15:15:00 | 1865.00 | 1862.36 | 1862.27 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1834.40 | 1856.77 | 1859.73 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 1873.50 | 1847.54 | 1844.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 13:15:00 | 1889.50 | 1862.96 | 1852.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 1882.55 | 1885.90 | 1871.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 1882.55 | 1885.90 | 1871.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1889.35 | 1899.71 | 1894.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 1889.35 | 1899.71 | 1894.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 1891.10 | 1897.99 | 1893.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 1871.00 | 1897.99 | 1893.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1851.20 | 1885.78 | 1888.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 1844.50 | 1872.64 | 1882.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1862.45 | 1859.18 | 1870.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1862.45 | 1859.18 | 1870.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1862.45 | 1859.18 | 1870.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 1866.60 | 1859.18 | 1870.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1857.05 | 1855.44 | 1863.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 1861.40 | 1855.44 | 1863.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1866.05 | 1857.55 | 1861.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 1866.05 | 1857.55 | 1861.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1872.55 | 1860.55 | 1862.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1872.55 | 1860.55 | 1862.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1885.25 | 1867.44 | 1865.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 1899.90 | 1884.40 | 1876.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 1892.70 | 1893.10 | 1886.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 10:15:00 | 1892.70 | 1893.10 | 1886.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1892.70 | 1893.10 | 1886.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 1886.65 | 1893.10 | 1886.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1908.35 | 1916.17 | 1910.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:30:00 | 1905.10 | 1916.17 | 1910.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1904.00 | 1913.74 | 1909.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:00:00 | 1904.00 | 1913.74 | 1909.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1905.55 | 1912.10 | 1909.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 13:00:00 | 1905.55 | 1912.10 | 1909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 1909.40 | 1910.31 | 1908.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1922.25 | 1910.24 | 1908.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 1935.50 | 1950.00 | 1951.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 1935.50 | 1950.00 | 1951.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1929.75 | 1938.85 | 1944.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 1921.90 | 1919.88 | 1929.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 1921.90 | 1919.88 | 1929.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1913.50 | 1911.96 | 1919.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 1909.15 | 1910.34 | 1918.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:30:00 | 1906.70 | 1899.84 | 1899.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 1932.65 | 1906.40 | 1902.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1932.65 | 1906.40 | 1902.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 1939.30 | 1923.06 | 1916.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 1953.65 | 1953.78 | 1941.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 1953.65 | 1953.78 | 1941.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1946.00 | 1951.07 | 1942.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 1962.95 | 1951.07 | 1942.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 11:15:00 | 1940.25 | 1949.20 | 1944.04 | SL hit (close<static) qty=1.00 sl=1941.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 1926.45 | 1941.44 | 1943.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 12:15:00 | 1921.25 | 1937.41 | 1941.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 1924.85 | 1916.11 | 1925.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 1924.85 | 1916.11 | 1925.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1924.85 | 1916.11 | 1925.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 1924.85 | 1916.11 | 1925.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 1935.05 | 1919.90 | 1926.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 1935.05 | 1919.90 | 1926.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 1932.90 | 1922.50 | 1926.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1924.00 | 1922.50 | 1926.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 11:15:00 | 1937.45 | 1928.54 | 1928.71 | SL hit (close>static) qty=1.00 sl=1935.15 alert=retest2 |

### Cycle 112 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 1941.60 | 1931.15 | 1929.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 10:15:00 | 1968.85 | 1941.36 | 1935.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 15:15:00 | 1975.00 | 1984.34 | 1971.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 15:15:00 | 1975.00 | 1984.34 | 1971.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 1975.00 | 1984.34 | 1971.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:15:00 | 1814.00 | 1984.34 | 1971.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2025-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 09:15:00 | 1810.80 | 1949.63 | 1956.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 10:15:00 | 1796.05 | 1819.66 | 1851.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 1789.00 | 1785.90 | 1807.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-17 15:00:00 | 1789.00 | 1785.90 | 1807.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1795.05 | 1786.77 | 1796.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 1796.30 | 1786.77 | 1796.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1794.95 | 1788.41 | 1796.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1811.00 | 1788.41 | 1796.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1810.35 | 1792.79 | 1797.90 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 12:15:00 | 1815.70 | 1802.11 | 1801.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 12:15:00 | 1821.60 | 1811.45 | 1806.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 10:15:00 | 1809.25 | 1817.25 | 1812.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 10:15:00 | 1809.25 | 1817.25 | 1812.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1809.25 | 1817.25 | 1812.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1809.25 | 1817.25 | 1812.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1819.20 | 1817.64 | 1812.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 1823.20 | 1818.27 | 1813.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 1804.50 | 1815.16 | 1812.91 | SL hit (close<static) qty=1.00 sl=1806.25 alert=retest2 |

### Cycle 115 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 1803.70 | 1811.55 | 1811.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 1792.00 | 1805.01 | 1808.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 10:15:00 | 1711.85 | 1711.34 | 1733.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 11:00:00 | 1711.85 | 1711.34 | 1733.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1726.55 | 1718.62 | 1727.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 1726.55 | 1718.62 | 1727.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1728.20 | 1720.54 | 1727.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:30:00 | 1713.25 | 1718.10 | 1725.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 1719.00 | 1718.01 | 1723.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 11:00:00 | 1718.65 | 1718.14 | 1722.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 11:30:00 | 1718.15 | 1717.90 | 1722.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 1725.00 | 1719.32 | 1722.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:00:00 | 1725.00 | 1719.32 | 1722.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1725.40 | 1720.54 | 1722.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:30:00 | 1728.10 | 1720.54 | 1722.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1725.90 | 1721.61 | 1723.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:45:00 | 1725.70 | 1721.61 | 1723.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 1724.00 | 1722.09 | 1723.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 1719.50 | 1722.09 | 1723.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1706.05 | 1695.42 | 1701.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 1706.05 | 1695.42 | 1701.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1706.45 | 1697.63 | 1701.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-04 14:15:00 | 1730.20 | 1708.24 | 1705.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 1730.20 | 1708.24 | 1705.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-11 09:15:00 | 1741.00 | 1730.27 | 1724.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 12:15:00 | 1716.95 | 1729.91 | 1726.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 12:15:00 | 1716.95 | 1729.91 | 1726.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 1716.95 | 1729.91 | 1726.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 13:00:00 | 1716.95 | 1729.91 | 1726.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 13:15:00 | 1721.10 | 1728.15 | 1725.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:00:00 | 1721.10 | 1728.15 | 1725.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 1722.90 | 1727.10 | 1725.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:30:00 | 1718.30 | 1727.10 | 1725.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 1723.90 | 1726.46 | 1725.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 1731.05 | 1726.46 | 1725.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 1720.85 | 1725.34 | 1725.02 | SL hit (close<static) qty=1.00 sl=1720.95 alert=retest2 |

### Cycle 117 — SELL (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 10:15:00 | 1705.10 | 1721.29 | 1723.21 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 14:15:00 | 1728.30 | 1714.55 | 1713.12 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 15:15:00 | 1707.70 | 1714.39 | 1714.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 09:15:00 | 1704.15 | 1712.34 | 1713.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 11:15:00 | 1696.80 | 1696.41 | 1702.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 11:45:00 | 1697.65 | 1696.41 | 1702.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 1702.60 | 1697.65 | 1702.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 1702.60 | 1697.65 | 1702.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1697.10 | 1697.54 | 1702.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 1656.05 | 1697.71 | 1701.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 1573.25 | 1616.26 | 1634.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1559.20 | 1549.78 | 1566.94 | SL hit (close>ema200) qty=0.50 sl=1549.78 alert=retest2 |

### Cycle 120 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 1589.30 | 1570.65 | 1570.65 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 1562.95 | 1572.51 | 1573.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 14:15:00 | 1557.70 | 1567.55 | 1571.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 1569.75 | 1566.76 | 1570.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1569.75 | 1566.76 | 1570.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1569.75 | 1566.76 | 1570.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1566.45 | 1566.76 | 1570.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1569.75 | 1567.36 | 1570.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1569.75 | 1567.36 | 1570.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1563.65 | 1566.61 | 1569.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 13:15:00 | 1559.70 | 1566.11 | 1568.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 1539.40 | 1558.78 | 1560.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1553.00 | 1539.97 | 1539.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1553.00 | 1539.97 | 1539.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1558.75 | 1550.68 | 1545.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1532.45 | 1548.68 | 1545.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 09:15:00 | 1532.45 | 1548.68 | 1545.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1532.45 | 1548.68 | 1545.52 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 11:15:00 | 1527.20 | 1541.82 | 1542.79 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 1568.30 | 1546.72 | 1544.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-21 11:15:00 | 1586.10 | 1568.98 | 1559.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 1571.60 | 1572.10 | 1563.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 13:15:00 | 1571.60 | 1572.10 | 1563.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 1571.60 | 1572.10 | 1563.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 1571.45 | 1572.10 | 1563.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1565.00 | 1569.49 | 1563.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 1570.00 | 1570.01 | 1564.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1634.00 | 1630.53 | 1619.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 15:15:00 | 1637.25 | 1626.72 | 1621.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1611.00 | 1625.26 | 1621.68 | SL hit (close<static) qty=1.00 sl=1613.55 alert=retest2 |

### Cycle 125 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 1597.50 | 1617.06 | 1618.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 1590.15 | 1611.68 | 1616.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1376.80 | 1372.95 | 1417.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1376.80 | 1372.95 | 1417.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1410.30 | 1383.94 | 1408.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1410.30 | 1383.94 | 1408.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1423.00 | 1391.75 | 1409.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 1423.00 | 1391.75 | 1409.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1413.05 | 1396.01 | 1410.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1375.15 | 1398.65 | 1408.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 1409.00 | 1401.82 | 1400.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1409.00 | 1401.82 | 1400.86 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 13:15:00 | 1390.00 | 1399.46 | 1399.87 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 1412.80 | 1402.24 | 1400.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 1416.30 | 1405.05 | 1402.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1386.00 | 1416.67 | 1413.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1386.00 | 1416.67 | 1413.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1386.00 | 1416.67 | 1413.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1394.00 | 1416.67 | 1413.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 1392.00 | 1411.73 | 1411.95 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 1417.10 | 1412.81 | 1412.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 12:15:00 | 1425.90 | 1415.42 | 1413.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 1476.30 | 1478.56 | 1462.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:30:00 | 1483.10 | 1478.56 | 1462.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1546.60 | 1574.54 | 1566.48 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 13:15:00 | 1552.00 | 1560.19 | 1561.27 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 1580.10 | 1563.27 | 1561.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 1588.70 | 1575.71 | 1571.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 11:15:00 | 1574.30 | 1575.74 | 1572.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 12:00:00 | 1574.30 | 1575.74 | 1572.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1575.30 | 1575.66 | 1572.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 14:30:00 | 1578.30 | 1574.48 | 1572.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 1562.40 | 1571.96 | 1572.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 1562.40 | 1571.96 | 1572.03 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 10:15:00 | 1572.60 | 1572.08 | 1572.08 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 1567.20 | 1571.11 | 1571.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 12:15:00 | 1556.00 | 1568.09 | 1570.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1576.90 | 1568.21 | 1569.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 1576.90 | 1568.21 | 1569.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1576.90 | 1568.21 | 1569.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 1581.40 | 1568.21 | 1569.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1574.10 | 1569.38 | 1569.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:15:00 | 1584.20 | 1569.38 | 1569.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 1597.70 | 1575.05 | 1572.34 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 14:15:00 | 1569.10 | 1572.37 | 1572.73 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1627.50 | 1583.02 | 1577.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1643.60 | 1601.88 | 1587.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1639.00 | 1640.15 | 1617.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 1636.50 | 1640.15 | 1617.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1616.10 | 1630.81 | 1619.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1616.10 | 1630.81 | 1619.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1620.60 | 1628.77 | 1620.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1633.40 | 1628.77 | 1620.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 1644.60 | 1652.68 | 1652.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 1644.60 | 1652.68 | 1652.96 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1656.10 | 1653.42 | 1653.14 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1648.30 | 1652.39 | 1652.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1644.00 | 1650.71 | 1651.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1647.00 | 1645.51 | 1648.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1647.00 | 1645.51 | 1648.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1637.80 | 1643.97 | 1647.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1621.20 | 1647.99 | 1648.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 15:00:00 | 1632.10 | 1636.03 | 1641.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1666.70 | 1642.80 | 1643.30 | SL hit (close>static) qty=1.00 sl=1650.80 alert=retest2 |

### Cycle 142 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1668.30 | 1647.90 | 1645.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1673.60 | 1654.50 | 1650.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1656.60 | 1665.30 | 1659.14 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1650.80 | 1656.48 | 1657.15 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1664.70 | 1657.64 | 1656.96 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1654.50 | 1657.80 | 1658.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1635.70 | 1653.38 | 1656.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1637.40 | 1632.69 | 1639.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 1637.40 | 1632.69 | 1639.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1631.90 | 1632.53 | 1638.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 1635.40 | 1632.53 | 1638.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1632.10 | 1632.19 | 1636.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 1625.60 | 1633.42 | 1635.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 1650.90 | 1636.66 | 1636.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1650.90 | 1636.66 | 1636.42 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1628.20 | 1635.09 | 1635.75 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1638.00 | 1634.79 | 1634.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1653.80 | 1638.59 | 1636.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1696.70 | 1704.20 | 1689.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 1696.70 | 1704.20 | 1689.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1694.90 | 1701.94 | 1692.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 1703.00 | 1701.94 | 1692.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 1697.40 | 1700.17 | 1695.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 1704.20 | 1717.45 | 1717.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1704.20 | 1717.45 | 1717.87 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1735.60 | 1718.39 | 1716.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1740.40 | 1727.38 | 1721.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1709.10 | 1725.25 | 1721.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1709.10 | 1725.25 | 1721.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1708.10 | 1721.82 | 1720.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1706.60 | 1721.82 | 1720.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 1713.40 | 1718.16 | 1718.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 1704.00 | 1714.25 | 1716.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1723.50 | 1713.80 | 1716.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 1710.80 | 1715.60 | 1716.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1710.60 | 1705.07 | 1710.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 1710.10 | 1706.66 | 1710.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:00:00 | 1710.60 | 1707.45 | 1710.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1719.20 | 1709.80 | 1711.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:00:00 | 1719.20 | 1709.80 | 1711.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1718.20 | 1711.48 | 1711.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:30:00 | 1717.90 | 1711.48 | 1711.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-25 14:15:00 | 1717.90 | 1712.76 | 1712.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1717.90 | 1712.76 | 1712.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 1721.40 | 1715.36 | 1713.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1722.50 | 1725.89 | 1721.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:15:00 | 1721.00 | 1725.89 | 1721.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1716.70 | 1724.05 | 1720.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:30:00 | 1716.70 | 1724.05 | 1720.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1724.80 | 1724.20 | 1721.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:30:00 | 1711.80 | 1724.20 | 1721.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1715.80 | 1723.24 | 1721.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1714.00 | 1723.24 | 1721.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1719.20 | 1722.43 | 1721.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1717.60 | 1722.43 | 1721.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1724.40 | 1722.83 | 1721.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 1725.90 | 1722.70 | 1721.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 1715.90 | 1724.29 | 1723.55 | SL hit (close<static) qty=1.00 sl=1720.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 1717.80 | 1722.99 | 1723.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 1712.70 | 1720.24 | 1721.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1721.10 | 1720.41 | 1721.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 1721.10 | 1720.41 | 1721.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1719.50 | 1720.23 | 1721.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 1712.70 | 1720.23 | 1721.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 1715.90 | 1719.69 | 1721.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 15:15:00 | 1715.00 | 1719.51 | 1720.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1722.10 | 1719.31 | 1720.46 | SL hit (close>static) qty=1.00 sl=1722.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1730.00 | 1721.44 | 1721.33 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 1718.70 | 1720.90 | 1721.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1714.50 | 1719.38 | 1720.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1718.80 | 1715.88 | 1718.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:00:00 | 1718.80 | 1715.88 | 1718.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1719.30 | 1716.56 | 1718.16 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1725.90 | 1719.90 | 1719.41 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1711.40 | 1719.16 | 1719.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1691.70 | 1708.38 | 1713.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1714.50 | 1707.03 | 1711.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1714.50 | 1707.03 | 1711.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1711.00 | 1707.82 | 1711.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1717.30 | 1707.82 | 1711.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1709.80 | 1708.22 | 1711.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1709.80 | 1708.22 | 1711.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1700.10 | 1706.60 | 1710.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1691.60 | 1706.60 | 1710.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1683.00 | 1701.88 | 1707.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 1680.30 | 1701.88 | 1707.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:00:00 | 1680.20 | 1697.54 | 1705.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 1679.40 | 1694.83 | 1703.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1596.28 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1596.19 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:15:00 | 1595.43 | 1608.00 | 1629.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 1549.20 | 1546.84 | 1557.90 | SL hit (close>ema200) qty=0.50 sl=1546.84 alert=retest2 |

### Cycle 158 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 1476.30 | 1466.89 | 1465.78 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 1463.30 | 1469.27 | 1469.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 1459.90 | 1466.55 | 1468.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1465.60 | 1463.09 | 1465.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1465.60 | 1463.09 | 1465.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1475.40 | 1465.55 | 1466.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1475.40 | 1465.55 | 1466.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1479.10 | 1468.26 | 1467.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 10:15:00 | 1484.30 | 1475.06 | 1471.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 1495.90 | 1497.62 | 1491.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 1495.90 | 1497.62 | 1491.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1499.30 | 1502.32 | 1497.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 1499.30 | 1502.32 | 1497.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1494.70 | 1500.80 | 1497.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 1494.70 | 1500.80 | 1497.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1489.20 | 1498.48 | 1496.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 1489.20 | 1498.48 | 1496.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1489.00 | 1496.58 | 1495.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 1481.90 | 1496.58 | 1495.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1485.30 | 1494.32 | 1495.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1474.20 | 1483.39 | 1488.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 1479.70 | 1479.18 | 1483.51 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1466.60 | 1479.18 | 1483.51 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1479.10 | 1477.46 | 1481.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:45:00 | 1477.60 | 1477.46 | 1481.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 1485.00 | 1478.96 | 1482.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 1485.00 | 1478.96 | 1482.15 | SL hit (close>ema400) qty=1.00 sl=1482.15 alert=retest1 |

### Cycle 162 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1497.40 | 1485.76 | 1484.73 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1472.90 | 1486.07 | 1487.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1466.00 | 1476.13 | 1481.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1491.80 | 1477.82 | 1481.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 1499.10 | 1477.82 | 1481.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1499.00 | 1482.05 | 1482.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1499.00 | 1482.05 | 1482.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1505.90 | 1486.82 | 1485.04 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1453.00 | 1486.59 | 1489.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1451.70 | 1471.97 | 1481.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1461.10 | 1459.81 | 1470.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1461.10 | 1459.81 | 1470.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1461.50 | 1457.66 | 1464.76 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1470.00 | 1466.69 | 1466.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 1473.40 | 1468.03 | 1467.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1462.60 | 1467.90 | 1467.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1462.60 | 1467.90 | 1467.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1465.20 | 1467.36 | 1467.16 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1465.00 | 1466.89 | 1466.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 09:15:00 | 1462.00 | 1465.91 | 1466.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1468.10 | 1463.00 | 1464.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1468.10 | 1463.00 | 1464.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1465.20 | 1463.44 | 1464.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1460.30 | 1463.44 | 1464.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1460.60 | 1430.92 | 1428.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1460.60 | 1430.92 | 1428.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1468.40 | 1438.41 | 1432.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 1464.60 | 1465.16 | 1458.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1464.60 | 1465.16 | 1458.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1455.70 | 1464.08 | 1460.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 1455.70 | 1464.08 | 1460.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1454.10 | 1462.09 | 1459.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 1454.10 | 1462.09 | 1459.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1464.90 | 1462.36 | 1460.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1462.80 | 1462.36 | 1460.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1472.10 | 1465.50 | 1462.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:30:00 | 1473.30 | 1467.86 | 1464.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 1474.70 | 1469.23 | 1465.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 1470.30 | 1481.05 | 1481.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1470.30 | 1481.05 | 1481.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1465.90 | 1476.70 | 1479.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1438.00 | 1434.82 | 1444.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 1438.00 | 1434.82 | 1444.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1441.10 | 1436.47 | 1443.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1442.40 | 1436.47 | 1443.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1441.30 | 1437.44 | 1443.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 1439.60 | 1438.17 | 1443.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1431.60 | 1439.32 | 1443.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1410.50 | 1392.78 | 1392.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1410.50 | 1392.78 | 1392.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 1414.60 | 1397.14 | 1394.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1487.50 | 1488.00 | 1475.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 14:15:00 | 1496.50 | 1488.27 | 1479.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 15:15:00 | 1494.10 | 1489.18 | 1480.49 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1492.00 | 1498.18 | 1491.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1504.50 | 1498.18 | 1491.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1497.50 | 1498.04 | 1492.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1497.70 | 1496.96 | 1492.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 1498.40 | 1496.64 | 1493.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1494.90 | 1496.29 | 1493.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 1494.60 | 1496.29 | 1493.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1497.60 | 1507.56 | 1502.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1497.60 | 1507.56 | 1502.56 | SL hit (close<ema400) qty=1.00 sl=1502.56 alert=retest1 |

### Cycle 171 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1489.30 | 1499.31 | 1499.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 1484.40 | 1496.33 | 1498.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1499.30 | 1495.10 | 1497.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 1503.40 | 1495.10 | 1497.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1494.90 | 1495.06 | 1497.16 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1507.00 | 1498.54 | 1498.44 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1494.90 | 1498.11 | 1498.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 1486.50 | 1495.66 | 1497.10 | Break + close below crossover candle low |

### Cycle 174 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1533.20 | 1503.17 | 1500.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1539.50 | 1510.43 | 1503.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1522.20 | 1524.00 | 1513.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 1522.20 | 1524.00 | 1513.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1519.90 | 1523.45 | 1517.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 1518.30 | 1523.45 | 1517.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1523.00 | 1523.36 | 1518.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 1516.90 | 1523.36 | 1518.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1530.30 | 1533.71 | 1528.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1530.30 | 1533.71 | 1528.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1519.20 | 1530.81 | 1527.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 1519.20 | 1530.81 | 1527.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1514.10 | 1527.47 | 1526.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1514.10 | 1527.47 | 1526.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 1515.50 | 1524.10 | 1525.07 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1533.40 | 1525.47 | 1525.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1548.00 | 1529.98 | 1527.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1545.00 | 1546.41 | 1538.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1545.00 | 1546.41 | 1538.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1546.40 | 1546.41 | 1539.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1541.70 | 1546.41 | 1539.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1548.10 | 1546.77 | 1541.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1541.60 | 1546.77 | 1541.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1551.60 | 1548.57 | 1543.43 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 1532.90 | 1541.85 | 1542.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 1527.40 | 1538.27 | 1540.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1530.00 | 1528.42 | 1532.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 1530.00 | 1528.42 | 1532.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1525.20 | 1527.46 | 1531.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:45:00 | 1530.30 | 1527.46 | 1531.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1532.00 | 1517.08 | 1521.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1532.00 | 1517.08 | 1521.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1541.80 | 1522.02 | 1523.17 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1537.00 | 1525.02 | 1524.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 1549.40 | 1536.51 | 1530.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1588.20 | 1594.02 | 1583.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1596.30 | 1594.48 | 1584.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1576.50 | 1590.50 | 1584.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1576.50 | 1590.50 | 1584.21 | SL hit (close<ema400) qty=1.00 sl=1584.21 alert=retest1 |

### Cycle 179 — SELL (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 13:15:00 | 1613.20 | 1631.55 | 1632.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1607.50 | 1626.74 | 1630.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1626.40 | 1623.80 | 1628.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:30:00 | 1630.40 | 1623.80 | 1628.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1625.90 | 1624.22 | 1627.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 1628.50 | 1624.22 | 1627.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1625.10 | 1624.40 | 1627.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:30:00 | 1630.20 | 1624.40 | 1627.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1623.00 | 1624.12 | 1627.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:00:00 | 1622.00 | 1623.69 | 1626.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1623.20 | 1616.53 | 1616.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1623.20 | 1616.53 | 1616.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 1629.60 | 1622.82 | 1620.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1622.80 | 1627.57 | 1624.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 1622.80 | 1627.57 | 1624.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1623.80 | 1626.82 | 1624.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1633.90 | 1626.82 | 1624.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 11:45:00 | 1632.00 | 1634.61 | 1631.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1658.20 | 1671.30 | 1671.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 1658.20 | 1671.30 | 1671.91 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 1672.40 | 1667.81 | 1667.53 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 09:15:00 | 1662.50 | 1666.75 | 1667.07 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 1671.20 | 1667.61 | 1667.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1672.90 | 1669.21 | 1668.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1667.00 | 1677.54 | 1674.44 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1648.40 | 1668.27 | 1670.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 1641.20 | 1655.92 | 1658.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1662.50 | 1648.34 | 1651.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1662.50 | 1648.34 | 1651.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1665.50 | 1651.77 | 1653.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1665.50 | 1651.77 | 1653.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 1665.00 | 1654.42 | 1654.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 1670.70 | 1657.67 | 1655.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 1674.30 | 1674.75 | 1668.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 1674.30 | 1674.75 | 1668.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1667.00 | 1673.32 | 1670.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1667.10 | 1673.32 | 1670.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1660.70 | 1670.80 | 1669.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1660.70 | 1670.80 | 1669.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1657.30 | 1668.10 | 1668.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1653.20 | 1661.69 | 1665.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1629.30 | 1625.35 | 1633.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1629.30 | 1625.35 | 1633.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1626.70 | 1625.04 | 1630.94 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 1642.80 | 1634.94 | 1634.32 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 1631.10 | 1633.96 | 1633.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1630.60 | 1633.29 | 1633.66 | Break + close below crossover candle low |

### Cycle 190 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1640.50 | 1634.73 | 1634.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1641.70 | 1637.49 | 1635.85 | Break + close above crossover candle high |

### Cycle 191 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 1599.70 | 1630.33 | 1632.90 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1642.00 | 1621.24 | 1621.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 11:15:00 | 1644.30 | 1628.90 | 1624.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 1641.10 | 1646.89 | 1640.42 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1659.60 | 1646.89 | 1640.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1650.20 | 1658.45 | 1651.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1650.20 | 1658.45 | 1651.97 | SL hit (close<ema400) qty=1.00 sl=1651.97 alert=retest1 |

### Cycle 193 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 1678.60 | 1689.35 | 1690.17 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1698.50 | 1687.38 | 1687.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1704.00 | 1693.40 | 1690.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1701.20 | 1704.38 | 1698.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 14:00:00 | 1701.20 | 1704.38 | 1698.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1706.70 | 1704.84 | 1698.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:30:00 | 1716.30 | 1707.28 | 1701.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 1717.00 | 1708.25 | 1702.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:45:00 | 1714.70 | 1709.06 | 1703.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1720.00 | 1714.21 | 1708.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1703.10 | 1716.64 | 1712.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1703.10 | 1716.64 | 1712.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1710.00 | 1715.31 | 1712.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 1701.10 | 1715.31 | 1712.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1694.30 | 1713.28 | 1713.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1694.30 | 1713.28 | 1713.08 | SL hit (close<static) qty=1.00 sl=1698.80 alert=retest2 |

### Cycle 195 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 1697.00 | 1710.03 | 1711.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 1686.00 | 1696.32 | 1702.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 1684.30 | 1681.03 | 1689.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:45:00 | 1686.40 | 1681.03 | 1689.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1710.90 | 1685.33 | 1688.37 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1718.00 | 1695.89 | 1692.89 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 1610.70 | 1680.02 | 1687.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1579.30 | 1591.15 | 1599.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1462.00 | 1458.11 | 1477.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1462.00 | 1458.11 | 1477.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1489.20 | 1465.52 | 1477.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1489.20 | 1465.52 | 1477.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1489.30 | 1470.28 | 1478.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1486.20 | 1470.28 | 1478.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1483.50 | 1482.86 | 1482.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1474.00 | 1482.86 | 1482.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1490.80 | 1472.50 | 1474.92 | SL hit (close>static) qty=1.00 sl=1484.80 alert=retest2 |

### Cycle 198 — BUY (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 13:15:00 | 1391.20 | 1386.64 | 1386.11 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1383.60 | 1385.78 | 1385.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 09:15:00 | 1346.80 | 1364.30 | 1371.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1351.80 | 1350.49 | 1360.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 1356.90 | 1350.49 | 1360.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1369.00 | 1355.03 | 1360.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1341.00 | 1359.55 | 1361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 13:00:00 | 1353.00 | 1357.75 | 1359.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 13:45:00 | 1352.70 | 1356.76 | 1359.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1352.60 | 1356.93 | 1358.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1362.20 | 1357.98 | 1359.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1351.50 | 1357.98 | 1359.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1352.30 | 1356.84 | 1358.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1363.00 | 1358.11 | 1358.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 1363.00 | 1358.11 | 1358.08 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 1356.80 | 1357.84 | 1357.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1348.90 | 1355.68 | 1356.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1361.30 | 1355.13 | 1356.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 1361.30 | 1355.13 | 1356.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1360.10 | 1356.13 | 1356.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 1362.50 | 1356.13 | 1356.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1357.00 | 1356.30 | 1356.61 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 1358.60 | 1357.05 | 1356.92 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1343.30 | 1354.29 | 1355.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1332.80 | 1348.24 | 1352.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1329.40 | 1323.05 | 1333.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1333.00 | 1323.05 | 1333.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1366.60 | 1331.09 | 1331.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1373.80 | 1339.63 | 1334.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1336.80 | 1351.53 | 1345.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1336.80 | 1351.53 | 1345.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1326.90 | 1346.61 | 1343.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1326.90 | 1346.61 | 1343.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1324.00 | 1339.33 | 1340.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1322.80 | 1336.03 | 1338.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1346.40 | 1331.94 | 1335.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1346.40 | 1331.94 | 1335.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1349.00 | 1335.35 | 1336.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1351.50 | 1335.35 | 1336.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 1340.50 | 1338.00 | 1337.93 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 1330.40 | 1337.67 | 1337.86 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 1339.90 | 1338.11 | 1338.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-23 10:15:00 | 1349.80 | 1341.71 | 1339.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1380.90 | 1387.24 | 1375.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 1380.90 | 1387.24 | 1375.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1389.70 | 1387.47 | 1377.60 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 1360.10 | 1374.79 | 1375.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1346.60 | 1369.15 | 1372.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1380.60 | 1357.66 | 1362.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 1363.00 | 1364.33 | 1365.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 1362.80 | 1359.67 | 1361.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 1378.20 | 1363.38 | 1363.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1378.20 | 1363.38 | 1363.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1399.40 | 1370.58 | 1366.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1439.40 | 1446.82 | 1432.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 1439.20 | 1446.82 | 1432.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1433.70 | 1451.48 | 1442.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 1439.30 | 1451.48 | 1442.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1429.30 | 1447.04 | 1441.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 1429.30 | 1447.04 | 1441.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1450.80 | 1444.16 | 1441.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 1443.40 | 1444.16 | 1441.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1427.70 | 1442.12 | 1440.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1426.00 | 1442.12 | 1440.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1423.40 | 1438.38 | 1439.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 11:15:00 | 1422.90 | 1435.28 | 1437.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1455.00 | 1435.70 | 1436.48 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1448.70 | 1438.30 | 1437.59 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 1433.20 | 1440.55 | 1441.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 1429.60 | 1434.93 | 1437.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1437.00 | 1433.66 | 1436.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 1437.00 | 1433.66 | 1436.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1442.30 | 1435.39 | 1436.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 1442.30 | 1435.39 | 1436.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 1449.40 | 1438.19 | 1438.03 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 1436.60 | 1437.87 | 1437.90 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1442.00 | 1438.70 | 1438.27 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1305.20 | 1412.41 | 1426.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 1299.80 | 1389.88 | 1414.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 1220.30 | 1220.00 | 1254.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 1220.30 | 1220.00 | 1254.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1207.10 | 1201.48 | 1209.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:45:00 | 1206.20 | 1201.48 | 1209.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1210.00 | 1203.18 | 1209.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 1210.00 | 1203.18 | 1209.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1198.10 | 1202.17 | 1208.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1209.50 | 1202.17 | 1208.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1196.70 | 1200.95 | 1206.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 1193.30 | 1199.24 | 1204.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:00:00 | 1194.40 | 1199.11 | 1201.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1193.50 | 1190.38 | 1192.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 1199.40 | 1193.15 | 1193.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 1199.40 | 1193.15 | 1193.06 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-12 10:15:00 | 1092.30 | 2023-05-17 09:15:00 | 1080.75 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-05-24 15:00:00 | 1114.10 | 2023-06-02 13:15:00 | 1137.60 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2023-05-25 12:30:00 | 1113.85 | 2023-06-02 13:15:00 | 1137.60 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2023-05-25 14:15:00 | 1113.80 | 2023-06-02 13:15:00 | 1137.60 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2023-05-25 14:45:00 | 1115.85 | 2023-06-02 13:15:00 | 1137.60 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2023-05-30 09:15:00 | 1142.45 | 2023-06-02 13:15:00 | 1137.60 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-06-09 14:15:00 | 1115.35 | 2023-06-12 10:15:00 | 1135.50 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2023-06-14 09:15:00 | 1132.50 | 2023-06-14 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-06-14 13:15:00 | 1134.25 | 2023-06-20 09:15:00 | 1136.90 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2023-06-15 09:15:00 | 1143.75 | 2023-06-20 09:15:00 | 1136.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-23 11:15:00 | 1161.00 | 2023-07-06 14:15:00 | 1180.30 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2023-07-27 11:15:00 | 1109.80 | 2023-07-27 12:15:00 | 1118.65 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-07-28 10:45:00 | 1107.55 | 2023-07-31 15:15:00 | 1115.65 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-07-31 11:00:00 | 1109.90 | 2023-07-31 15:15:00 | 1115.65 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-07-31 11:45:00 | 1110.05 | 2023-07-31 15:15:00 | 1115.65 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-08-03 09:30:00 | 1133.60 | 2023-08-03 10:15:00 | 1121.25 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-08-04 09:15:00 | 1139.10 | 2023-08-09 09:15:00 | 1128.85 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2023-08-29 14:30:00 | 1157.50 | 2023-08-30 09:15:00 | 1167.15 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-09-01 10:00:00 | 1177.95 | 2023-09-15 09:15:00 | 1295.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-28 11:15:00 | 1249.00 | 2023-10-06 11:15:00 | 1251.85 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-10-03 12:45:00 | 1249.50 | 2023-10-06 11:15:00 | 1251.85 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2023-10-05 11:15:00 | 1242.15 | 2023-10-06 11:15:00 | 1251.85 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-10-10 09:15:00 | 1251.95 | 2023-10-11 14:15:00 | 1244.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-11-02 09:15:00 | 1272.35 | 2023-11-06 13:15:00 | 1266.25 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-11-02 10:30:00 | 1271.25 | 2023-11-06 13:15:00 | 1266.25 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2023-11-02 11:45:00 | 1271.40 | 2023-11-06 13:15:00 | 1266.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2023-11-06 09:15:00 | 1274.55 | 2023-11-06 13:15:00 | 1266.25 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-08 13:15:00 | 1276.65 | 2023-11-08 14:15:00 | 1267.40 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-11-13 11:00:00 | 1256.00 | 2023-11-15 09:15:00 | 1281.65 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2023-11-23 09:15:00 | 1332.60 | 2023-11-24 09:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-11-23 12:45:00 | 1333.80 | 2023-11-24 09:15:00 | 1318.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-12-04 14:30:00 | 1343.00 | 2023-12-05 09:15:00 | 1318.50 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2023-12-14 09:15:00 | 1412.20 | 2023-12-20 14:15:00 | 1441.95 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2024-01-12 09:15:00 | 1509.35 | 2024-01-20 11:15:00 | 1561.20 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2024-01-23 12:00:00 | 1541.95 | 2024-01-24 14:15:00 | 1578.80 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-01-23 13:00:00 | 1542.05 | 2024-01-24 14:15:00 | 1578.80 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-01-24 10:00:00 | 1544.30 | 2024-01-24 14:15:00 | 1578.80 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-01-24 10:30:00 | 1540.05 | 2024-01-24 14:15:00 | 1578.80 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-02-01 09:30:00 | 1578.95 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-02-01 13:30:00 | 1577.00 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-02-01 14:30:00 | 1576.65 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-02-01 15:00:00 | 1577.00 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-02-02 09:15:00 | 1598.50 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-02-02 14:00:00 | 1581.65 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-02-02 14:30:00 | 1582.80 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-02-02 15:00:00 | 1583.50 | 2024-02-05 12:15:00 | 1570.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-02-09 13:30:00 | 1632.25 | 2024-02-20 10:15:00 | 1646.30 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-02-09 15:00:00 | 1635.50 | 2024-02-20 10:15:00 | 1646.30 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-02-12 09:15:00 | 1655.25 | 2024-02-20 10:15:00 | 1646.30 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-03-18 09:15:00 | 1657.10 | 2024-03-18 15:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-03-18 11:00:00 | 1651.05 | 2024-03-18 15:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-03-18 14:15:00 | 1652.55 | 2024-03-18 15:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-03-21 11:45:00 | 1596.80 | 2024-03-22 09:15:00 | 1516.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-21 14:45:00 | 1594.30 | 2024-03-22 09:15:00 | 1514.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-21 11:45:00 | 1596.80 | 2024-03-26 14:15:00 | 1563.35 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2024-03-21 14:45:00 | 1594.30 | 2024-03-26 14:15:00 | 1563.35 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2024-03-22 09:15:00 | 1515.00 | 2024-04-08 09:15:00 | 1546.30 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-04-10 09:15:00 | 1547.15 | 2024-04-10 13:15:00 | 1537.85 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-04-10 10:00:00 | 1543.90 | 2024-04-10 13:15:00 | 1537.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-04-22 13:30:00 | 1457.75 | 2024-04-23 09:15:00 | 1484.90 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-04-25 09:15:00 | 1492.65 | 2024-04-26 14:15:00 | 1471.05 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest1 | 2024-05-07 09:15:00 | 1334.60 | 2024-05-09 11:15:00 | 1334.30 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-05-09 15:00:00 | 1317.20 | 2024-05-15 09:15:00 | 1321.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-05-14 15:00:00 | 1319.70 | 2024-05-15 09:15:00 | 1321.75 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-05-28 13:30:00 | 1362.65 | 2024-05-30 09:15:00 | 1336.45 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-05-29 13:15:00 | 1360.10 | 2024-05-30 09:15:00 | 1336.45 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1320.35 | 2024-06-04 12:15:00 | 1254.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1320.35 | 2024-06-04 14:15:00 | 1312.45 | STOP_HIT | 0.50 | 0.60% |
| BUY | retest2 | 2024-06-20 11:30:00 | 1441.85 | 2024-06-24 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-06-20 14:15:00 | 1441.70 | 2024-06-24 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-06-24 11:00:00 | 1440.90 | 2024-06-24 11:15:00 | 1440.20 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-06-25 12:15:00 | 1440.10 | 2024-06-25 14:15:00 | 1449.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-06-25 13:15:00 | 1439.75 | 2024-06-25 14:15:00 | 1449.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-07-02 09:15:00 | 1480.95 | 2024-07-10 09:15:00 | 1509.15 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-07-08 12:30:00 | 1518.95 | 2024-07-10 10:15:00 | 1492.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-07-18 13:00:00 | 1585.50 | 2024-07-22 14:15:00 | 1576.05 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-07-22 09:45:00 | 1588.30 | 2024-07-22 14:15:00 | 1576.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-07-30 14:15:00 | 1627.95 | 2024-08-02 09:15:00 | 1618.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-07-31 09:45:00 | 1627.20 | 2024-08-02 09:15:00 | 1618.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1582.50 | 2024-08-07 15:15:00 | 1598.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-06 14:30:00 | 1583.00 | 2024-08-07 15:15:00 | 1598.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-06 15:00:00 | 1580.70 | 2024-08-07 15:15:00 | 1598.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-13 11:00:00 | 1607.25 | 2024-08-23 11:15:00 | 1669.10 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1614.55 | 2024-08-23 11:15:00 | 1669.10 | STOP_HIT | 1.00 | 3.38% |
| BUY | retest2 | 2024-08-29 10:15:00 | 1747.15 | 2024-09-05 10:15:00 | 1773.00 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2024-08-29 12:00:00 | 1743.00 | 2024-09-05 10:15:00 | 1773.00 | STOP_HIT | 1.00 | 1.72% |
| BUY | retest2 | 2024-08-29 14:45:00 | 1747.30 | 2024-09-05 10:15:00 | 1773.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1795.00 | 2024-09-18 09:15:00 | 1772.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-09-12 14:00:00 | 1793.50 | 2024-09-18 09:15:00 | 1772.05 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-25 13:15:00 | 1764.50 | 2024-10-03 11:15:00 | 1780.55 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2024-10-14 12:30:00 | 1854.20 | 2024-10-18 11:15:00 | 1848.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-10-14 13:30:00 | 1854.00 | 2024-10-18 11:15:00 | 1848.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-10-18 10:30:00 | 1853.55 | 2024-10-18 11:15:00 | 1848.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-10-24 10:30:00 | 1838.90 | 2024-10-24 11:15:00 | 1852.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2024-10-31 09:15:00 | 1816.25 | 2024-11-06 09:15:00 | 1821.85 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-11-14 09:15:00 | 1895.90 | 2024-11-14 10:15:00 | 1855.90 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1922.25 | 2024-12-19 10:15:00 | 1935.50 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-12-24 11:30:00 | 1909.15 | 2024-12-30 14:15:00 | 1932.65 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-12-30 13:30:00 | 1906.70 | 2024-12-30 14:15:00 | 1932.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-01-06 09:15:00 | 1962.95 | 2025-01-06 11:15:00 | 1940.25 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-01-06 12:30:00 | 1949.95 | 2025-01-07 10:15:00 | 1928.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1969.25 | 2025-01-07 10:15:00 | 1928.15 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1924.00 | 2025-01-09 11:15:00 | 1937.45 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-01-23 12:30:00 | 1823.20 | 2025-01-23 14:15:00 | 1804.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-30 13:30:00 | 1713.25 | 2025-02-04 14:15:00 | 1730.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-01-31 10:15:00 | 1719.00 | 2025-02-04 14:15:00 | 1730.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-01-31 11:00:00 | 1718.65 | 2025-02-04 14:15:00 | 1730.20 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-31 11:30:00 | 1718.15 | 2025-02-04 14:15:00 | 1730.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-02-12 09:15:00 | 1731.05 | 2025-02-12 09:15:00 | 1720.85 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-24 09:15:00 | 1656.05 | 2025-02-28 09:15:00 | 1573.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 1656.05 | 2025-03-05 09:15:00 | 1559.20 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2025-03-10 13:15:00 | 1559.70 | 2025-03-18 09:15:00 | 1553.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-03-12 09:15:00 | 1539.40 | 2025-03-18 09:15:00 | 1553.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-03-27 15:15:00 | 1637.25 | 2025-03-28 09:15:00 | 1611.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1375.15 | 2025-04-11 12:15:00 | 1409.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-05-06 14:30:00 | 1578.30 | 2025-05-07 09:15:00 | 1562.40 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1633.40 | 2025-05-19 13:15:00 | 1644.60 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1621.20 | 2025-05-23 09:15:00 | 1666.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-05-22 15:00:00 | 1632.10 | 2025-05-23 09:15:00 | 1666.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-06-04 15:00:00 | 1625.60 | 2025-06-05 11:15:00 | 1650.90 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-06-13 10:15:00 | 1703.00 | 2025-06-19 09:15:00 | 1704.20 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-06-13 15:15:00 | 1697.40 | 2025-06-19 09:15:00 | 1704.20 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-06-24 13:15:00 | 1710.80 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1710.60 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-06-25 10:45:00 | 1710.10 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-25 12:00:00 | 1710.60 | 2025-06-25 14:15:00 | 1717.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-30 13:15:00 | 1725.90 | 2025-07-01 12:15:00 | 1715.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-02 13:15:00 | 1712.70 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-02 14:15:00 | 1715.90 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-02 15:15:00 | 1715.00 | 2025-07-03 09:15:00 | 1722.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-09 10:15:00 | 1680.30 | 2025-07-15 09:15:00 | 1596.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 11:00:00 | 1680.20 | 2025-07-15 09:15:00 | 1596.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 12:15:00 | 1679.40 | 2025-07-15 09:15:00 | 1595.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-09 10:15:00 | 1680.30 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2025-07-09 11:00:00 | 1680.20 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.80% |
| SELL | retest2 | 2025-07-09 12:15:00 | 1679.40 | 2025-07-18 14:15:00 | 1549.20 | STOP_HIT | 0.50 | 7.75% |
| SELL | retest1 | 2025-08-20 09:15:00 | 1466.60 | 2025-08-20 11:15:00 | 1485.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1460.30 | 2025-09-10 09:15:00 | 1460.60 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-09-16 12:30:00 | 1473.30 | 2025-09-19 12:15:00 | 1470.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-09-16 14:00:00 | 1474.70 | 2025-09-19 12:15:00 | 1470.30 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-09-24 14:15:00 | 1439.60 | 2025-10-06 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 2.02% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1431.60 | 2025-10-06 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest1 | 2025-10-13 14:15:00 | 1496.50 | 2025-10-17 09:15:00 | 1497.60 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest1 | 2025-10-13 15:15:00 | 1494.10 | 2025-10-17 09:15:00 | 1497.60 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1504.50 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1497.50 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-15 12:15:00 | 1497.70 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-15 14:15:00 | 1498.40 | 2025-10-17 13:15:00 | 1489.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-11-14 11:00:00 | 1596.30 | 2025-11-14 12:15:00 | 1576.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-14 14:30:00 | 1585.10 | 2025-11-21 13:15:00 | 1613.20 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2025-11-24 14:00:00 | 1622.00 | 2025-11-26 13:15:00 | 1623.20 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1633.90 | 2025-12-09 14:15:00 | 1658.20 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-12-02 11:45:00 | 1632.00 | 2025-12-09 14:15:00 | 1658.20 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest1 | 2026-01-09 09:15:00 | 1659.60 | 2026-01-12 09:15:00 | 1650.20 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-12 12:00:00 | 1655.00 | 2026-01-13 09:15:00 | 1643.30 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-12 13:30:00 | 1655.70 | 2026-01-13 09:15:00 | 1643.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-13 10:30:00 | 1656.80 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2026-01-13 14:15:00 | 1658.60 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2026-01-14 11:15:00 | 1669.70 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2026-01-14 14:30:00 | 1670.00 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2026-01-14 15:00:00 | 1668.30 | 2026-01-21 09:15:00 | 1678.60 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2026-01-27 09:30:00 | 1716.30 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-01-27 11:30:00 | 1717.00 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-27 12:45:00 | 1714.70 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-28 09:30:00 | 1720.00 | 2026-01-30 09:15:00 | 1694.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-02-18 09:15:00 | 1474.00 | 2026-02-19 09:15:00 | 1490.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1480.00 | 2026-02-24 09:15:00 | 1406.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1480.00 | 2026-02-24 12:15:00 | 1332.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1341.00 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-03-09 13:00:00 | 1353.00 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-09 13:45:00 | 1352.70 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-03-09 14:30:00 | 1352.60 | 2026-03-11 11:15:00 | 1363.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-04-01 13:00:00 | 1363.00 | 2026-04-02 12:15:00 | 1378.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-04-02 12:00:00 | 1362.80 | 2026-04-02 12:15:00 | 1378.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-05-04 13:15:00 | 1193.30 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-05-06 12:00:00 | 1194.40 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-08 11:45:00 | 1193.50 | 2026-05-08 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.49% |
