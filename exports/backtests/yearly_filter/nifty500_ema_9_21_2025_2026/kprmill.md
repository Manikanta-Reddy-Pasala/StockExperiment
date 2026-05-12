# K.P.R. Mill Ltd. (KPRMILL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 86 |
| ALERT1 | 49 |
| ALERT2 | 47 |
| ALERT2_SKIP | 31 |
| ALERT3 | 124 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 43 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 37
- **Target hits / Stop hits / Partials:** 3 / 43 / 2
- **Avg / median % per leg:** -0.12% / -0.79%
- **Sum % (uncompounded):** -5.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 3 | 14.3% | 1 | 20 | 0 | -0.30% | -6.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 3 | 14.3% | 1 | 20 | 0 | -0.30% | -6.4% |
| SELL (all) | 27 | 8 | 29.6% | 2 | 23 | 2 | 0.03% | 0.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 8 | 29.6% | 2 | 23 | 2 | 0.03% | 0.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 11 | 22.9% | 3 | 43 | 2 | -0.12% | -5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 11:15:00 | 1150.85 | 1176.74 | 1179.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 14:15:00 | 1146.80 | 1163.09 | 1172.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 1157.65 | 1155.52 | 1162.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 15:00:00 | 1157.65 | 1155.52 | 1162.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1150.05 | 1146.47 | 1152.64 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1183.00 | 1159.97 | 1157.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1207.35 | 1173.91 | 1164.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 1159.85 | 1212.67 | 1203.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1159.85 | 1212.67 | 1203.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1159.85 | 1212.67 | 1203.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:00:00 | 1179.20 | 1199.32 | 1198.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 1189.50 | 1197.35 | 1197.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 1189.50 | 1197.35 | 1197.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 11:15:00 | 1206.75 | 1197.94 | 1197.43 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1175.00 | 1193.66 | 1195.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 1160.90 | 1187.11 | 1192.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 11:15:00 | 1157.85 | 1156.42 | 1169.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:30:00 | 1162.00 | 1156.42 | 1169.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1153.05 | 1155.10 | 1163.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1122.70 | 1144.39 | 1149.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 15:15:00 | 1138.90 | 1132.47 | 1131.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1138.90 | 1132.47 | 1131.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 1141.20 | 1134.13 | 1132.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 1137.50 | 1139.80 | 1136.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1137.50 | 1139.80 | 1136.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1137.50 | 1139.80 | 1136.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1136.60 | 1139.80 | 1136.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1150.00 | 1141.84 | 1137.86 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1130.20 | 1136.25 | 1136.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 10:15:00 | 1125.10 | 1134.02 | 1135.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 14:15:00 | 1126.60 | 1121.35 | 1123.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 14:15:00 | 1126.60 | 1121.35 | 1123.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1126.60 | 1121.35 | 1123.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:15:00 | 1126.00 | 1121.35 | 1123.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1126.00 | 1122.28 | 1124.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1121.10 | 1122.28 | 1124.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1110.60 | 1115.48 | 1119.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 15:00:00 | 1107.00 | 1113.78 | 1118.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 1105.60 | 1101.52 | 1102.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 13:15:00 | 1108.00 | 1102.32 | 1102.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 1108.00 | 1102.32 | 1102.28 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1101.00 | 1102.18 | 1102.27 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 12:15:00 | 1108.40 | 1103.18 | 1102.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 1112.90 | 1105.62 | 1103.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1100.00 | 1106.63 | 1105.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 1100.00 | 1106.63 | 1105.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1100.00 | 1106.63 | 1105.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1098.80 | 1106.63 | 1105.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1108.70 | 1107.04 | 1105.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 1100.10 | 1107.04 | 1105.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1115.80 | 1108.79 | 1106.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:15:00 | 1105.00 | 1108.79 | 1106.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1105.00 | 1108.04 | 1106.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 1092.10 | 1108.04 | 1106.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 1093.70 | 1105.17 | 1105.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 15:15:00 | 1074.80 | 1092.90 | 1098.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1095.70 | 1093.46 | 1098.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1095.70 | 1093.46 | 1098.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1095.70 | 1093.46 | 1098.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1095.70 | 1093.46 | 1098.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1100.40 | 1094.85 | 1098.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 1100.40 | 1094.85 | 1098.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1099.40 | 1095.76 | 1098.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 1097.70 | 1095.76 | 1098.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1101.60 | 1096.93 | 1098.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 1101.60 | 1096.93 | 1098.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1103.70 | 1098.28 | 1099.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 1101.10 | 1098.28 | 1099.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1127.60 | 1104.14 | 1101.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1133.50 | 1119.89 | 1110.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 13:15:00 | 1120.70 | 1125.67 | 1120.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 1120.70 | 1125.67 | 1120.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1120.70 | 1125.67 | 1120.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 1120.70 | 1125.67 | 1120.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1129.50 | 1126.44 | 1121.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:45:00 | 1135.80 | 1125.10 | 1122.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:30:00 | 1132.50 | 1136.02 | 1131.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1130.40 | 1136.94 | 1132.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:00:00 | 1134.50 | 1136.75 | 1133.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1141.80 | 1137.76 | 1133.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 1141.20 | 1137.76 | 1133.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1104.20 | 1131.45 | 1131.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1104.20 | 1131.45 | 1131.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1100.60 | 1125.28 | 1128.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 1113.90 | 1096.91 | 1106.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 1113.90 | 1096.91 | 1106.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1113.90 | 1096.91 | 1106.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 1113.90 | 1096.91 | 1106.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1109.00 | 1099.33 | 1107.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 1115.00 | 1106.38 | 1109.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1144.60 | 1114.03 | 1112.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 1165.50 | 1124.32 | 1117.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 1172.10 | 1174.47 | 1156.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 10:00:00 | 1172.10 | 1174.47 | 1156.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1146.90 | 1165.65 | 1156.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1146.80 | 1165.65 | 1156.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1145.90 | 1161.70 | 1155.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:45:00 | 1144.60 | 1161.70 | 1155.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1157.90 | 1160.69 | 1156.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1176.60 | 1160.69 | 1156.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 1173.90 | 1186.96 | 1188.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1173.90 | 1186.96 | 1188.39 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1189.00 | 1188.19 | 1188.14 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1184.60 | 1187.47 | 1187.82 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1189.50 | 1188.08 | 1188.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1196.60 | 1189.74 | 1188.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 1198.40 | 1199.16 | 1194.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:00:00 | 1198.40 | 1199.16 | 1194.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1183.10 | 1195.95 | 1193.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 1183.90 | 1195.95 | 1193.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1182.30 | 1193.22 | 1192.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1199.10 | 1193.22 | 1192.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 1184.70 | 1198.94 | 1199.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1184.70 | 1198.94 | 1199.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 1179.70 | 1195.09 | 1197.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 1189.20 | 1188.59 | 1191.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 15:15:00 | 1189.20 | 1188.59 | 1191.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1189.20 | 1188.59 | 1191.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1186.90 | 1188.59 | 1191.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1183.60 | 1187.59 | 1191.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 1178.00 | 1187.59 | 1191.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:30:00 | 1181.80 | 1184.80 | 1188.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1180.70 | 1184.80 | 1188.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 1202.50 | 1188.34 | 1189.73 | SL hit (close>static) qty=1.00 sl=1199.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 1208.00 | 1192.27 | 1191.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 1218.00 | 1197.42 | 1193.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1214.30 | 1226.74 | 1216.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 13:15:00 | 1214.30 | 1226.74 | 1216.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1214.30 | 1226.74 | 1216.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1217.60 | 1226.74 | 1216.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1245.40 | 1230.47 | 1219.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 1202.70 | 1230.47 | 1219.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1207.00 | 1227.94 | 1220.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 1211.80 | 1227.94 | 1220.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1198.90 | 1222.13 | 1218.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 1199.30 | 1222.13 | 1218.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 1199.20 | 1213.69 | 1214.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 1191.20 | 1209.19 | 1212.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 1179.90 | 1170.88 | 1186.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 1179.90 | 1170.88 | 1186.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1179.60 | 1160.08 | 1167.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 1179.60 | 1160.08 | 1167.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1182.80 | 1164.62 | 1168.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 1176.90 | 1167.08 | 1169.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 09:15:00 | 1118.06 | 1164.21 | 1167.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-04 09:15:00 | 1059.21 | 1079.75 | 1111.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 22 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 995.70 | 973.37 | 970.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1016.70 | 996.17 | 987.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 1018.40 | 1018.68 | 1009.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 1018.40 | 1018.68 | 1009.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1021.00 | 1019.91 | 1012.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1009.70 | 1019.91 | 1012.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1028.00 | 1025.93 | 1019.66 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 991.30 | 1016.50 | 1018.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 981.50 | 1004.75 | 1012.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 987.50 | 980.96 | 988.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 987.50 | 980.96 | 988.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 987.50 | 980.96 | 988.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 987.50 | 980.96 | 988.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 978.80 | 980.53 | 988.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 988.50 | 980.53 | 988.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 988.50 | 982.18 | 987.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:00:00 | 988.50 | 982.18 | 987.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 985.70 | 982.88 | 987.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:30:00 | 988.10 | 982.88 | 987.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 985.90 | 983.49 | 987.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 981.00 | 983.49 | 987.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 988.60 | 984.51 | 987.32 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 997.50 | 988.29 | 988.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1017.55 | 994.14 | 990.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 996.05 | 997.40 | 993.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 996.05 | 997.40 | 993.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 992.90 | 997.80 | 995.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 993.10 | 997.80 | 995.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 990.00 | 996.24 | 994.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 989.75 | 996.24 | 994.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 995.80 | 995.02 | 994.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 999.90 | 995.02 | 994.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 998.10 | 996.05 | 994.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:30:00 | 998.70 | 996.00 | 995.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 982.95 | 993.39 | 994.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 982.95 | 993.39 | 994.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 975.95 | 988.14 | 991.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 988.00 | 986.62 | 990.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:00:00 | 988.00 | 986.62 | 990.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 983.15 | 985.93 | 989.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 988.95 | 985.93 | 989.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 988.10 | 986.36 | 989.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 999.10 | 990.38 | 990.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 1003.70 | 993.04 | 992.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 15:15:00 | 1009.90 | 1002.87 | 997.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 1002.75 | 1005.91 | 1001.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 1002.75 | 1005.91 | 1001.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1002.75 | 1005.91 | 1001.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 1000.95 | 1005.91 | 1001.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1000.00 | 1004.73 | 1001.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 1000.00 | 1004.73 | 1001.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 995.95 | 1002.97 | 1000.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:45:00 | 995.55 | 1002.97 | 1000.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 998.50 | 1002.08 | 1000.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1017.20 | 1002.08 | 1000.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 13:45:00 | 998.75 | 1002.63 | 1001.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 994.80 | 1001.06 | 1001.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 994.80 | 1001.06 | 1001.09 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1006.65 | 1001.37 | 1001.18 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 999.75 | 1001.05 | 1001.10 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1009.85 | 1002.10 | 1001.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 11:15:00 | 1042.00 | 1019.84 | 1012.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1096.60 | 1100.29 | 1085.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1096.60 | 1100.29 | 1085.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1131.00 | 1114.49 | 1102.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1131.00 | 1114.49 | 1102.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1109.70 | 1114.55 | 1107.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 1109.55 | 1114.55 | 1107.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 1105.65 | 1112.77 | 1107.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 1105.65 | 1112.77 | 1107.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1139.70 | 1118.16 | 1110.51 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 1105.25 | 1111.14 | 1111.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 1100.20 | 1108.95 | 1110.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 1114.20 | 1069.07 | 1075.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 14:15:00 | 1114.20 | 1069.07 | 1075.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1114.20 | 1069.07 | 1075.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1114.20 | 1069.07 | 1075.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1102.00 | 1075.65 | 1077.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1097.75 | 1075.65 | 1077.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1095.15 | 1079.55 | 1079.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 1095.15 | 1079.55 | 1079.51 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 1050.00 | 1076.58 | 1078.75 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 1064.40 | 1063.31 | 1063.27 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 1056.40 | 1062.53 | 1062.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 1048.40 | 1059.50 | 1061.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 1002.50 | 995.07 | 1001.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1002.50 | 995.07 | 1001.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1002.50 | 995.07 | 1001.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1002.50 | 995.07 | 1001.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 993.70 | 994.79 | 1000.95 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 13:15:00 | 1015.20 | 1004.36 | 1004.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1031.90 | 1020.78 | 1014.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1039.10 | 1040.78 | 1031.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 1039.10 | 1040.78 | 1031.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1039.40 | 1040.51 | 1032.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1039.80 | 1040.51 | 1032.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1021.00 | 1036.04 | 1031.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 1021.00 | 1036.04 | 1031.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1025.50 | 1033.94 | 1031.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 1027.00 | 1032.33 | 1030.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 1027.80 | 1032.33 | 1030.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 1026.60 | 1030.34 | 1029.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1025.70 | 1029.41 | 1029.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1025.70 | 1029.41 | 1029.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 1023.00 | 1027.38 | 1028.55 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1075.00 | 1036.91 | 1032.77 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1045.00 | 1055.23 | 1055.80 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1062.10 | 1056.60 | 1056.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1077.50 | 1060.78 | 1058.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1065.00 | 1067.29 | 1062.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1065.00 | 1067.29 | 1062.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1065.00 | 1067.29 | 1062.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1065.00 | 1067.29 | 1062.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1066.50 | 1067.13 | 1062.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1064.00 | 1067.13 | 1062.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1065.10 | 1073.48 | 1071.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1065.10 | 1073.48 | 1071.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1067.90 | 1072.37 | 1070.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1080.00 | 1072.37 | 1070.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1078.00 | 1075.94 | 1074.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 1062.60 | 1072.72 | 1073.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1062.60 | 1072.72 | 1073.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1050.50 | 1068.28 | 1070.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1060.90 | 1056.51 | 1062.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 1060.90 | 1056.51 | 1062.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1060.90 | 1056.51 | 1062.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 1060.90 | 1056.51 | 1062.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1067.50 | 1058.71 | 1062.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 1067.50 | 1058.71 | 1062.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1065.60 | 1060.09 | 1062.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1075.10 | 1060.09 | 1062.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 15:15:00 | 1084.00 | 1064.87 | 1064.77 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 1055.80 | 1063.06 | 1063.95 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 1075.70 | 1066.23 | 1065.21 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 1049.50 | 1064.81 | 1065.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 14:15:00 | 1045.30 | 1054.36 | 1059.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 1079.00 | 1043.94 | 1048.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 14:15:00 | 1079.00 | 1043.94 | 1048.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1079.00 | 1043.94 | 1048.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 1079.00 | 1043.94 | 1048.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1120.00 | 1059.15 | 1055.43 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 1085.00 | 1096.49 | 1097.82 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1102.90 | 1095.74 | 1095.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 1110.00 | 1098.81 | 1096.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1087.00 | 1097.92 | 1096.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1087.00 | 1097.92 | 1096.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1087.00 | 1097.92 | 1096.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1087.00 | 1097.92 | 1096.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1077.20 | 1093.77 | 1095.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 1063.90 | 1077.96 | 1081.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 983.10 | 982.86 | 994.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 992.30 | 982.86 | 994.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 991.30 | 984.55 | 994.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 980.80 | 984.51 | 992.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:00:00 | 979.00 | 977.26 | 979.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:15:00 | 981.60 | 972.56 | 974.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 12:15:00 | 969.00 | 966.25 | 966.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 969.00 | 966.25 | 966.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 975.10 | 968.63 | 967.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 966.00 | 968.11 | 967.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 15:15:00 | 966.00 | 968.11 | 967.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 966.00 | 968.11 | 967.13 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 963.70 | 967.23 | 967.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 959.00 | 965.03 | 966.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 944.20 | 942.89 | 950.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 944.20 | 942.89 | 950.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 946.20 | 943.55 | 949.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 951.40 | 943.55 | 949.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 948.10 | 944.46 | 949.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 942.60 | 944.94 | 948.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 942.60 | 944.94 | 948.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 950.00 | 945.49 | 947.31 | SL hit (close>static) qty=1.00 sl=949.80 alert=retest2 |

### Cycle 52 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 953.00 | 948.49 | 948.45 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 940.40 | 946.88 | 947.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 931.10 | 943.72 | 946.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 14:15:00 | 943.60 | 926.44 | 930.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 14:15:00 | 943.60 | 926.44 | 930.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 943.60 | 926.44 | 930.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 943.60 | 926.44 | 930.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 941.40 | 929.43 | 931.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 931.60 | 929.43 | 931.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 14:15:00 | 885.02 | 897.16 | 902.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 09:15:00 | 838.44 | 858.58 | 876.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 852.45 | 844.53 | 843.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 857.75 | 848.23 | 845.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 853.05 | 854.66 | 850.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 853.05 | 854.66 | 850.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 853.05 | 854.66 | 850.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 853.05 | 854.66 | 850.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 845.50 | 852.83 | 850.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 845.50 | 852.83 | 850.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 844.00 | 851.06 | 849.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:00:00 | 844.00 | 851.06 | 849.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 13:15:00 | 842.75 | 848.28 | 848.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 839.00 | 846.42 | 847.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 830.80 | 819.24 | 826.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 830.80 | 819.24 | 826.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 830.80 | 819.24 | 826.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 830.80 | 819.24 | 826.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 833.60 | 822.12 | 826.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:00:00 | 833.60 | 822.12 | 826.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 851.00 | 831.56 | 830.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 15:15:00 | 880.00 | 846.55 | 838.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 12:15:00 | 855.40 | 857.66 | 847.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:45:00 | 855.55 | 857.66 | 847.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 854.05 | 857.07 | 852.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 854.05 | 857.07 | 852.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 850.80 | 855.81 | 852.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 850.80 | 855.81 | 852.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 853.00 | 855.25 | 852.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 856.25 | 855.25 | 852.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-28 09:15:00 | 941.88 | 897.46 | 878.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 860.60 | 885.81 | 889.18 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 891.95 | 883.41 | 882.38 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 865.50 | 880.36 | 881.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 855.75 | 875.44 | 879.01 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1001.30 | 891.42 | 882.88 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 951.35 | 966.84 | 967.67 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 15:15:00 | 977.05 | 968.92 | 968.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 994.25 | 973.99 | 970.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 12:15:00 | 976.80 | 983.57 | 976.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 12:15:00 | 976.80 | 983.57 | 976.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 976.80 | 983.57 | 976.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 976.80 | 983.57 | 976.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 983.00 | 983.45 | 977.27 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 10:15:00 | 950.45 | 970.75 | 972.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 927.60 | 951.18 | 961.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 910.95 | 909.20 | 926.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 910.95 | 909.20 | 926.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 898.80 | 907.12 | 924.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 883.90 | 905.05 | 921.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 984.25 | 914.16 | 919.33 | SL hit (close>static) qty=1.00 sl=927.90 alert=retest2 |

### Cycle 64 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 926.40 | 922.62 | 922.50 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 913.00 | 920.70 | 921.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 904.45 | 915.76 | 919.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 918.55 | 913.75 | 916.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 918.55 | 913.75 | 916.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 918.55 | 913.75 | 916.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 918.55 | 913.75 | 916.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 914.00 | 913.80 | 916.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:30:00 | 911.60 | 912.93 | 915.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:45:00 | 912.25 | 913.36 | 915.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 912.20 | 912.22 | 914.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 911.70 | 908.21 | 909.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 910.50 | 908.67 | 909.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:15:00 | 913.00 | 908.67 | 909.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 913.00 | 909.53 | 910.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 900.50 | 909.53 | 910.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 10:15:00 | 916.35 | 910.45 | 910.51 | SL hit (close>static) qty=1.00 sl=915.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 918.15 | 911.99 | 911.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 12:15:00 | 922.80 | 914.15 | 912.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 918.80 | 925.64 | 921.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 918.80 | 925.64 | 921.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 918.80 | 925.64 | 921.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 919.50 | 925.64 | 921.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 920.95 | 924.70 | 921.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:15:00 | 918.80 | 924.70 | 921.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 918.80 | 923.52 | 920.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 914.75 | 923.52 | 920.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 917.50 | 922.32 | 920.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 908.20 | 922.32 | 920.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 900.00 | 916.75 | 918.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 897.90 | 911.43 | 915.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 917.45 | 912.64 | 915.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 14:15:00 | 917.45 | 912.64 | 915.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 917.45 | 912.64 | 915.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 15:00:00 | 917.45 | 912.64 | 915.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 915.00 | 913.11 | 915.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 908.80 | 913.11 | 915.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 911.45 | 912.40 | 913.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 910.80 | 912.58 | 913.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 910.25 | 912.53 | 913.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 913.30 | 912.69 | 913.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 914.60 | 912.69 | 913.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 919.00 | 913.95 | 913.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 919.00 | 913.95 | 913.61 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 905.45 | 912.25 | 912.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 896.40 | 909.08 | 911.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 842.00 | 840.17 | 851.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 09:30:00 | 840.05 | 840.17 | 851.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 835.00 | 830.35 | 838.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:30:00 | 833.85 | 830.35 | 838.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 831.50 | 830.58 | 837.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 14:30:00 | 836.10 | 830.58 | 837.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 837.50 | 832.37 | 836.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:30:00 | 838.30 | 832.37 | 836.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 837.15 | 833.32 | 836.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:30:00 | 838.00 | 833.32 | 836.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 837.00 | 834.06 | 836.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 837.00 | 834.06 | 836.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 833.00 | 833.85 | 836.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 837.30 | 833.85 | 836.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 828.25 | 832.47 | 835.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 826.75 | 832.47 | 835.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 863.20 | 827.94 | 828.59 | SL hit (close>static) qty=1.00 sl=838.90 alert=retest2 |

### Cycle 70 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 867.20 | 835.79 | 832.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 14:15:00 | 906.65 | 854.88 | 841.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 834.15 | 854.22 | 843.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 834.15 | 854.22 | 843.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 834.15 | 854.22 | 843.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 834.15 | 854.22 | 843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 831.50 | 849.68 | 842.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 832.95 | 849.68 | 842.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 819.65 | 841.36 | 840.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 819.65 | 841.36 | 840.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 14:15:00 | 836.45 | 838.88 | 839.07 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 15:15:00 | 850.00 | 841.10 | 840.06 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 822.30 | 837.34 | 838.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 811.30 | 832.13 | 835.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 826.80 | 826.46 | 831.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 826.80 | 826.46 | 831.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 826.80 | 826.46 | 831.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 833.85 | 826.46 | 831.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 829.05 | 826.74 | 830.80 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 851.35 | 833.24 | 832.16 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 826.00 | 833.27 | 833.71 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 12:15:00 | 841.85 | 834.86 | 834.35 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 832.65 | 834.08 | 834.14 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 838.85 | 835.04 | 834.57 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 821.50 | 831.75 | 833.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 804.25 | 824.88 | 829.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 11:15:00 | 825.55 | 822.79 | 827.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 11:15:00 | 825.55 | 822.79 | 827.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 825.55 | 822.79 | 827.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 825.55 | 822.79 | 827.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 819.60 | 822.15 | 827.02 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 836.75 | 827.77 | 827.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 13:15:00 | 844.75 | 831.17 | 829.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 845.00 | 847.46 | 840.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 845.00 | 847.46 | 840.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 845.00 | 847.46 | 840.82 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 816.80 | 837.67 | 839.13 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 845.35 | 833.10 | 832.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 862.25 | 844.12 | 839.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 851.40 | 856.34 | 849.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 851.40 | 856.34 | 849.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 851.40 | 856.34 | 849.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 851.40 | 856.34 | 849.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 858.60 | 856.79 | 850.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 861.00 | 856.79 | 850.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 920.00 | 923.17 | 923.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 920.00 | 923.17 | 923.55 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 12:15:00 | 930.15 | 924.56 | 924.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 936.75 | 929.39 | 926.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 11:15:00 | 930.30 | 930.48 | 927.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:45:00 | 929.45 | 930.48 | 927.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 928.75 | 931.66 | 929.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 928.75 | 931.66 | 929.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 932.15 | 931.76 | 929.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 14:15:00 | 934.00 | 931.92 | 930.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 933.80 | 932.29 | 930.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 926.50 | 930.93 | 930.30 | SL hit (close<static) qty=1.00 sl=927.00 alert=retest2 |

### Cycle 85 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 925.00 | 929.99 | 930.16 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 934.85 | 930.26 | 930.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 937.40 | 932.97 | 931.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 950.30 | 950.66 | 944.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 954.40 | 953.33 | 949.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 954.40 | 953.33 | 949.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 954.40 | 953.33 | 949.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 964.65 | 968.08 | 962.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 965.95 | 968.08 | 962.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 962.00 | 966.86 | 962.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 962.00 | 966.86 | 962.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 960.75 | 965.64 | 961.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 960.75 | 965.64 | 961.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 957.15 | 963.94 | 961.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 957.15 | 963.94 | 961.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 12:00:00 | 1179.20 | 2025-05-21 12:15:00 | 1189.50 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-05-30 15:00:00 | 1122.70 | 2025-06-04 15:15:00 | 1138.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1107.00 | 2025-06-17 13:15:00 | 1108.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-17 11:15:00 | 1105.60 | 2025-06-17 13:15:00 | 1108.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-06-27 09:45:00 | 1135.80 | 2025-07-01 09:15:00 | 1104.20 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-06-30 09:30:00 | 1132.50 | 2025-07-01 09:15:00 | 1104.20 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-06-30 10:30:00 | 1130.40 | 2025-07-01 09:15:00 | 1104.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-06-30 14:00:00 | 1134.50 | 2025-07-01 09:15:00 | 1104.20 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1176.60 | 2025-07-11 11:15:00 | 1173.90 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-07-16 09:15:00 | 1199.10 | 2025-07-18 13:15:00 | 1184.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-22 10:15:00 | 1178.00 | 2025-07-22 14:15:00 | 1202.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-22 13:30:00 | 1181.80 | 2025-07-22 14:15:00 | 1202.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1180.70 | 2025-07-22 14:15:00 | 1202.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1176.90 | 2025-07-31 09:15:00 | 1118.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:00:00 | 1176.90 | 2025-08-04 09:15:00 | 1059.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-03 15:15:00 | 999.90 | 2025-09-04 15:15:00 | 982.95 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-09-04 10:15:00 | 998.10 | 2025-09-04 15:15:00 | 982.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-04 14:30:00 | 998.70 | 2025-09-04 15:15:00 | 982.95 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1017.20 | 2025-09-10 14:15:00 | 994.80 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-09-10 13:45:00 | 998.75 | 2025-09-10 14:15:00 | 994.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1097.75 | 2025-09-30 09:15:00 | 1095.15 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-10-20 11:30:00 | 1027.00 | 2025-10-20 14:15:00 | 1025.70 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-10-20 12:15:00 | 1027.80 | 2025-10-20 14:15:00 | 1025.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-10-20 14:15:00 | 1026.60 | 2025-10-20 14:15:00 | 1025.70 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1080.00 | 2025-11-04 10:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1078.00 | 2025-11-04 10:15:00 | 1062.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-08 12:00:00 | 980.80 | 2025-12-19 12:15:00 | 969.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2025-12-10 15:00:00 | 979.00 | 2025-12-19 12:15:00 | 969.00 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-12-12 12:15:00 | 981.60 | 2025-12-19 12:15:00 | 969.00 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-12-26 15:00:00 | 942.60 | 2025-12-29 13:15:00 | 950.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-29 09:15:00 | 942.60 | 2025-12-29 13:15:00 | 950.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-01 09:15:00 | 931.60 | 2026-01-07 14:15:00 | 885.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:15:00 | 931.60 | 2026-01-09 09:15:00 | 838.44 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-27 09:15:00 | 856.25 | 2026-01-28 09:15:00 | 941.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 883.90 | 2026-02-13 12:15:00 | 984.25 | STOP_HIT | 1.00 | -11.35% |
| SELL | retest2 | 2026-02-17 11:30:00 | 911.60 | 2026-02-20 10:15:00 | 916.35 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-02-17 14:45:00 | 912.25 | 2026-02-20 11:15:00 | 918.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-18 09:30:00 | 912.20 | 2026-02-20 11:15:00 | 918.15 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-02-19 13:30:00 | 911.70 | 2026-02-20 11:15:00 | 918.15 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-02-20 09:15:00 | 900.50 | 2026-02-20 11:15:00 | 918.15 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-25 09:15:00 | 908.80 | 2026-02-26 15:15:00 | 919.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-26 09:15:00 | 911.45 | 2026-02-26 15:15:00 | 919.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-02-26 11:30:00 | 910.80 | 2026-02-26 15:15:00 | 919.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-26 14:15:00 | 910.25 | 2026-02-26 15:15:00 | 919.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-03-11 10:15:00 | 826.75 | 2026-03-12 11:15:00 | 863.20 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2026-04-09 11:15:00 | 861.00 | 2026-04-24 11:15:00 | 920.00 | STOP_HIT | 1.00 | 6.85% |
| BUY | retest2 | 2026-04-28 14:15:00 | 934.00 | 2026-04-29 09:15:00 | 926.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-04-28 15:00:00 | 933.80 | 2026-04-29 09:15:00 | 926.50 | STOP_HIT | 1.00 | -0.78% |
