# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2748.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 32 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 64 |
| PARTIAL | 3 |
| TARGET_HIT | 10 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 29 / 37
- **Target hits / Stop hits / Partials:** 10 / 53 / 3
- **Avg / median % per leg:** 1.01% / -0.47%
- **Sum % (uncompounded):** 66.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 17 | 48.6% | 7 | 28 | 0 | 1.69% | 59.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 17 | 48.6% | 7 | 28 | 0 | 1.69% | 59.1% |
| SELL (all) | 31 | 12 | 38.7% | 3 | 25 | 3 | 0.25% | 7.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| SELL @ 3rd Alert (retest2) | 30 | 12 | 40.0% | 3 | 24 | 3 | 0.28% | 8.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.47% | -0.5% |
| retest2 (combined) | 65 | 29 | 44.6% | 10 | 52 | 3 | 1.04% | 67.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1219.60 | 1169.39 | 1164.21 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 1173.40 | 1178.01 | 1178.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 1155.80 | 1172.80 | 1175.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1151.20 | 1133.51 | 1142.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:15:00 | 1179.70 | 1133.51 | 1142.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1188.60 | 1144.53 | 1146.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 1188.60 | 1144.53 | 1146.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1178.00 | 1151.22 | 1149.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 1160.40 | 1180.58 | 1180.66 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1197.00 | 1183.05 | 1181.44 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1170.10 | 1182.95 | 1183.51 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1184.10 | 1181.73 | 1181.71 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1181.00 | 1181.59 | 1181.65 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1185.50 | 1182.28 | 1181.94 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 11:15:00 | 1179.90 | 1181.42 | 1181.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1175.20 | 1179.15 | 1180.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1178.00 | 1177.32 | 1178.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1166.80 | 1177.32 | 1178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1162.40 | 1174.34 | 1177.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:15:00 | 1157.00 | 1174.34 | 1177.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1160.70 | 1171.61 | 1175.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1155.80 | 1137.42 | 1134.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1155.80 | 1137.42 | 1134.96 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1118.60 | 1136.00 | 1136.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 1111.20 | 1122.56 | 1128.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1126.90 | 1121.79 | 1125.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1126.90 | 1121.79 | 1125.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1121.90 | 1121.81 | 1125.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1127.10 | 1121.81 | 1125.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1125.50 | 1122.55 | 1125.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 1118.80 | 1122.14 | 1124.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1116.10 | 1120.93 | 1123.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1119.20 | 1113.93 | 1118.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:30:00 | 1118.40 | 1115.73 | 1118.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1106.60 | 1103.97 | 1109.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1106.60 | 1103.97 | 1109.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1099.50 | 1103.08 | 1108.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:15:00 | 1109.80 | 1103.08 | 1108.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1109.80 | 1104.42 | 1109.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1091.60 | 1104.42 | 1109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1075.80 | 1098.70 | 1105.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 1108.50 | 1099.81 | 1099.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 1108.50 | 1099.81 | 1099.06 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1090.80 | 1097.30 | 1098.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1081.20 | 1094.08 | 1096.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1101.00 | 1094.72 | 1096.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 1101.00 | 1094.72 | 1096.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1081.20 | 1092.01 | 1095.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1102.70 | 1094.15 | 1095.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1102.40 | 1095.80 | 1096.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1103.00 | 1095.80 | 1096.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1100.50 | 1097.27 | 1096.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1134.30 | 1104.81 | 1100.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1096.40 | 1110.04 | 1103.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1099.00 | 1110.04 | 1103.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1096.90 | 1107.41 | 1103.34 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 13:15:00 | 1092.50 | 1101.10 | 1101.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 1091.30 | 1095.86 | 1098.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1094.70 | 1094.43 | 1096.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 1098.30 | 1094.43 | 1096.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1099.00 | 1095.35 | 1097.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:45:00 | 1095.40 | 1095.35 | 1097.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1094.40 | 1095.16 | 1096.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:30:00 | 1092.80 | 1094.95 | 1096.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 1092.70 | 1094.76 | 1096.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 1100.00 | 1087.40 | 1090.23 | SL hit (close>static) qty=1.00 sl=1099.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 1100.20 | 1093.10 | 1092.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1131.90 | 1100.86 | 1096.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 1141.90 | 1142.85 | 1127.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 1141.90 | 1142.85 | 1127.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1132.50 | 1142.32 | 1130.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 1126.00 | 1142.32 | 1130.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1123.70 | 1138.60 | 1129.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1123.70 | 1138.60 | 1129.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1133.10 | 1137.50 | 1129.89 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 1099.80 | 1122.32 | 1124.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 1095.00 | 1106.29 | 1114.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1110.70 | 1104.19 | 1110.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1112.50 | 1104.19 | 1110.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1104.40 | 1104.24 | 1109.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:45:00 | 1099.00 | 1102.87 | 1108.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1097.40 | 1103.29 | 1108.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 1120.00 | 1107.08 | 1108.95 | SL hit (close>static) qty=1.00 sl=1112.80 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 13:15:00 | 1119.30 | 1110.26 | 1110.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 1120.90 | 1112.39 | 1111.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1113.90 | 1124.35 | 1120.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1113.90 | 1124.35 | 1120.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1113.50 | 1122.18 | 1119.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1112.30 | 1122.18 | 1119.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 1107.60 | 1116.72 | 1117.57 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 1126.10 | 1117.44 | 1116.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 15:15:00 | 1130.10 | 1121.91 | 1119.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 1130.30 | 1130.55 | 1125.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 1129.70 | 1130.55 | 1125.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1127.40 | 1129.92 | 1126.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1127.40 | 1129.92 | 1126.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1132.00 | 1130.34 | 1126.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 1144.60 | 1132.40 | 1128.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1147.20 | 1132.40 | 1128.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 1159.00 | 1138.01 | 1132.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1188.70 | 1200.77 | 1202.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1188.70 | 1200.77 | 1202.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1181.60 | 1194.53 | 1198.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1195.90 | 1187.57 | 1192.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1195.90 | 1187.57 | 1192.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1201.80 | 1190.41 | 1193.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1200.70 | 1190.41 | 1193.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1187.30 | 1190.32 | 1192.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 15:00:00 | 1185.30 | 1189.05 | 1191.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1186.10 | 1189.24 | 1191.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1179.80 | 1187.45 | 1190.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1190.00 | 1161.63 | 1161.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1190.00 | 1161.63 | 1161.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1197.20 | 1173.89 | 1167.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1164.60 | 1175.17 | 1169.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:00:00 | 1164.60 | 1175.17 | 1169.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1222.40 | 1184.61 | 1173.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1247.50 | 1190.09 | 1177.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-01 09:15:00 | 1372.25 | 1293.15 | 1247.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1295.40 | 1310.46 | 1310.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 1286.20 | 1305.61 | 1308.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1300.00 | 1295.95 | 1302.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1288.50 | 1295.95 | 1302.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1291.20 | 1295.00 | 1301.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1270.80 | 1288.64 | 1295.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 1271.10 | 1282.14 | 1290.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1271.40 | 1282.14 | 1290.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:15:00 | 1273.20 | 1279.17 | 1287.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1293.00 | 1281.60 | 1286.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1309.20 | 1281.60 | 1286.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1300.40 | 1285.36 | 1288.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 11:15:00 | 1324.60 | 1295.65 | 1292.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 1324.60 | 1295.65 | 1292.52 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1290.50 | 1301.78 | 1303.20 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1341.50 | 1309.26 | 1306.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1353.80 | 1318.16 | 1310.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 1375.10 | 1376.52 | 1354.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 10:00:00 | 1375.10 | 1376.52 | 1354.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1422.00 | 1426.18 | 1414.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1407.20 | 1426.18 | 1414.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1409.10 | 1422.76 | 1414.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 1422.60 | 1415.91 | 1412.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:00:00 | 1423.20 | 1414.68 | 1413.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 14:15:00 | 1401.00 | 1411.25 | 1411.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1401.00 | 1411.25 | 1411.98 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1422.50 | 1411.67 | 1411.61 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 13:15:00 | 1410.00 | 1411.52 | 1411.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1405.30 | 1410.27 | 1411.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1414.70 | 1411.16 | 1411.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1402.70 | 1411.16 | 1411.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1412.70 | 1411.47 | 1411.46 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 10:15:00 | 1410.00 | 1411.17 | 1411.32 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1415.90 | 1412.12 | 1411.74 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 1402.00 | 1410.64 | 1411.23 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1430.10 | 1411.85 | 1410.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1461.10 | 1427.54 | 1419.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 1462.30 | 1465.62 | 1447.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 1462.30 | 1465.62 | 1447.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1474.50 | 1493.83 | 1480.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1474.50 | 1493.83 | 1480.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1471.30 | 1489.33 | 1479.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 1462.10 | 1489.33 | 1479.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1472.90 | 1486.04 | 1479.15 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1457.00 | 1475.36 | 1475.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 12:15:00 | 1446.20 | 1465.19 | 1469.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1458.00 | 1451.26 | 1459.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 1458.00 | 1451.26 | 1459.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1456.30 | 1452.36 | 1458.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 1456.30 | 1452.36 | 1458.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1456.40 | 1453.17 | 1458.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1464.60 | 1453.17 | 1458.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1455.00 | 1453.53 | 1457.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1473.50 | 1453.53 | 1457.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1467.50 | 1456.33 | 1458.69 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1461.90 | 1459.82 | 1459.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 1474.30 | 1462.72 | 1460.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 1467.20 | 1468.48 | 1465.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:30:00 | 1467.00 | 1468.48 | 1465.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1466.00 | 1467.58 | 1465.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1466.00 | 1467.58 | 1465.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1467.00 | 1467.47 | 1465.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1465.60 | 1467.47 | 1465.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1461.10 | 1474.61 | 1471.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1461.10 | 1474.61 | 1471.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1475.00 | 1474.69 | 1471.72 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1460.80 | 1468.39 | 1469.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 10:15:00 | 1437.50 | 1455.96 | 1462.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1461.00 | 1456.41 | 1461.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 1463.50 | 1456.41 | 1461.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1454.50 | 1456.03 | 1460.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 1453.50 | 1454.62 | 1459.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 1453.50 | 1455.46 | 1459.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 1449.10 | 1453.86 | 1456.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1380.83 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1380.83 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1376.64 | 1412.73 | 1429.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 14:15:00 | 1308.15 | 1335.13 | 1355.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1385.00 | 1359.81 | 1357.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1392.50 | 1370.86 | 1363.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1377.40 | 1382.13 | 1374.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 1377.40 | 1382.13 | 1374.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1389.60 | 1383.62 | 1375.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 1378.70 | 1383.62 | 1375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1399.40 | 1394.48 | 1384.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:00:00 | 1406.00 | 1396.79 | 1386.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1425.00 | 1400.78 | 1392.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 1407.00 | 1403.72 | 1395.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 1408.00 | 1400.77 | 1397.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1480.60 | 1487.21 | 1471.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 1488.40 | 1486.76 | 1472.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 1494.40 | 1498.39 | 1481.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-15 10:15:00 | 1546.60 | 1506.71 | 1486.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1700.30 | 1713.95 | 1714.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 1689.50 | 1709.06 | 1712.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1712.20 | 1706.32 | 1710.22 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1752.90 | 1715.38 | 1712.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1777.20 | 1735.78 | 1725.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1819.30 | 1823.60 | 1797.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 1784.50 | 1823.60 | 1797.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1769.50 | 1812.78 | 1794.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1769.50 | 1812.78 | 1794.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1778.60 | 1805.94 | 1793.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1763.60 | 1805.94 | 1793.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1759.40 | 1784.66 | 1785.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 10:15:00 | 1753.50 | 1770.16 | 1776.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1805.00 | 1767.01 | 1770.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1805.00 | 1767.01 | 1770.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1786.60 | 1770.93 | 1771.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1805.80 | 1770.93 | 1771.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 1790.60 | 1774.86 | 1773.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 1796.10 | 1779.11 | 1775.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 14:15:00 | 1781.30 | 1782.38 | 1777.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 1781.30 | 1782.38 | 1777.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1750.00 | 1775.90 | 1775.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1693.20 | 1775.90 | 1775.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1688.30 | 1758.38 | 1767.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 10:15:00 | 1677.30 | 1742.17 | 1759.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 1708.70 | 1708.25 | 1726.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 11:45:00 | 1712.00 | 1708.25 | 1726.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1704.00 | 1699.46 | 1709.21 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1729.50 | 1715.63 | 1714.14 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 1707.20 | 1713.13 | 1713.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1690.00 | 1708.00 | 1710.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 1707.80 | 1707.56 | 1709.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 12:30:00 | 1707.60 | 1707.56 | 1709.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1711.10 | 1708.26 | 1710.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1711.10 | 1708.26 | 1710.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1709.40 | 1708.49 | 1710.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 1709.40 | 1708.49 | 1710.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1714.90 | 1709.77 | 1710.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 1705.10 | 1709.77 | 1710.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1720.10 | 1711.84 | 1711.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1720.10 | 1711.84 | 1711.32 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 1701.10 | 1711.16 | 1711.19 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 1716.90 | 1712.12 | 1711.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 1726.10 | 1716.56 | 1713.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1713.90 | 1716.03 | 1713.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1713.90 | 1716.03 | 1713.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1717.60 | 1716.34 | 1714.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 1727.10 | 1718.21 | 1715.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1706.90 | 1720.44 | 1718.39 | SL hit (close<static) qty=1.00 sl=1710.10 alert=retest2 |

### Cycle 50 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1697.50 | 1715.42 | 1716.66 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 1756.00 | 1721.62 | 1718.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 11:15:00 | 1837.80 | 1744.86 | 1729.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1805.00 | 1807.77 | 1770.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:45:00 | 1826.70 | 1807.77 | 1770.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1771.00 | 1787.49 | 1777.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1773.00 | 1787.49 | 1777.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1786.60 | 1787.31 | 1778.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 1772.40 | 1787.31 | 1778.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1793.30 | 1788.38 | 1781.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:30:00 | 1772.60 | 1788.38 | 1781.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1783.00 | 1787.24 | 1782.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:15:00 | 1780.80 | 1787.24 | 1782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1767.30 | 1783.25 | 1780.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1767.30 | 1783.25 | 1780.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1788.10 | 1784.22 | 1781.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:30:00 | 1792.60 | 1786.52 | 1782.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1792.10 | 1786.75 | 1783.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 1792.10 | 1793.85 | 1787.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1769.00 | 1801.45 | 1803.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 1769.00 | 1801.45 | 1803.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 1754.90 | 1786.69 | 1795.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 1711.00 | 1707.66 | 1722.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1693.50 | 1707.66 | 1722.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1683.90 | 1665.78 | 1687.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1683.90 | 1665.78 | 1687.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1686.20 | 1669.86 | 1686.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:45:00 | 1687.90 | 1669.86 | 1686.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1686.00 | 1673.09 | 1686.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:00:00 | 1675.40 | 1673.55 | 1685.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1701.50 | 1676.50 | 1683.87 | SL hit (close>ema400) qty=1.00 sl=1683.87 alert=retest1 |

### Cycle 53 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1696.50 | 1689.13 | 1688.30 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1679.80 | 1686.28 | 1687.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 11:15:00 | 1666.50 | 1678.61 | 1683.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1667.30 | 1666.97 | 1674.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 1667.40 | 1666.97 | 1674.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1659.90 | 1665.56 | 1673.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1654.00 | 1663.38 | 1671.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1656.40 | 1662.75 | 1670.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1648.60 | 1663.12 | 1668.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1675.00 | 1653.78 | 1655.41 | SL hit (close>static) qty=1.00 sl=1673.90 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1671.30 | 1657.28 | 1656.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1688.30 | 1663.49 | 1659.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1672.70 | 1674.83 | 1666.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 1672.70 | 1674.83 | 1666.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1668.20 | 1673.51 | 1666.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 1665.00 | 1673.51 | 1666.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1664.90 | 1671.79 | 1666.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 1664.90 | 1671.79 | 1666.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1662.00 | 1669.83 | 1666.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 1660.50 | 1669.83 | 1666.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1651.30 | 1666.12 | 1664.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1651.30 | 1666.12 | 1664.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1660.10 | 1664.79 | 1664.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1640.20 | 1664.79 | 1664.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1630.10 | 1657.85 | 1661.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 12:15:00 | 1624.50 | 1633.06 | 1642.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1633.90 | 1633.23 | 1641.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1633.10 | 1633.23 | 1641.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1647.20 | 1636.02 | 1642.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1648.00 | 1636.02 | 1642.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1648.00 | 1638.42 | 1642.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1668.00 | 1638.42 | 1642.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1674.20 | 1650.60 | 1647.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 1681.40 | 1660.13 | 1652.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1668.00 | 1669.37 | 1660.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1656.00 | 1669.37 | 1660.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1657.70 | 1667.04 | 1660.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 1658.60 | 1667.04 | 1660.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1649.00 | 1663.43 | 1659.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1649.00 | 1663.43 | 1659.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1660.40 | 1671.86 | 1666.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1660.40 | 1671.86 | 1666.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1654.70 | 1668.42 | 1665.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1654.70 | 1668.42 | 1665.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 1656.00 | 1663.23 | 1663.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1645.40 | 1657.86 | 1660.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1661.20 | 1647.55 | 1653.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1661.20 | 1647.55 | 1653.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1650.00 | 1648.04 | 1653.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 1662.10 | 1648.04 | 1653.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1648.80 | 1648.19 | 1653.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1695.40 | 1648.19 | 1653.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1680.00 | 1654.55 | 1655.58 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 10:15:00 | 1678.40 | 1659.32 | 1657.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 1694.60 | 1666.38 | 1661.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 1670.90 | 1671.40 | 1664.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:45:00 | 1675.50 | 1671.40 | 1664.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1651.40 | 1667.40 | 1663.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1651.40 | 1667.40 | 1663.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1660.00 | 1665.92 | 1662.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1671.30 | 1665.92 | 1662.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 1663.50 | 1665.44 | 1663.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 1652.00 | 1660.56 | 1661.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1652.00 | 1660.56 | 1661.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 1644.50 | 1657.34 | 1659.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1660.00 | 1657.88 | 1659.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1660.00 | 1657.88 | 1659.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1658.00 | 1657.90 | 1659.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 1659.30 | 1657.32 | 1659.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1655.00 | 1656.86 | 1658.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1661.60 | 1656.86 | 1658.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1657.80 | 1657.05 | 1658.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1657.80 | 1657.05 | 1658.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1668.00 | 1659.24 | 1659.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1668.00 | 1659.24 | 1659.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1661.00 | 1659.59 | 1659.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:15:00 | 1680.00 | 1659.59 | 1659.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1705.10 | 1668.69 | 1663.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1733.00 | 1692.81 | 1677.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1749.30 | 1761.69 | 1747.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 1749.30 | 1761.69 | 1747.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1762.60 | 1761.87 | 1749.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 1770.90 | 1755.23 | 1748.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1773.00 | 1757.53 | 1752.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1770.00 | 1757.53 | 1752.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 1776.80 | 1767.76 | 1757.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1764.00 | 1767.82 | 1760.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1750.00 | 1767.82 | 1760.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1760.10 | 1766.28 | 1760.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 1750.30 | 1766.28 | 1760.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1748.00 | 1762.62 | 1759.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1748.00 | 1762.62 | 1759.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1754.00 | 1760.90 | 1758.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1754.00 | 1760.90 | 1758.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1754.40 | 1759.60 | 1758.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 1752.60 | 1759.60 | 1758.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 1739.00 | 1754.33 | 1756.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 1739.00 | 1754.33 | 1756.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 1732.00 | 1749.86 | 1753.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 1677.10 | 1676.62 | 1694.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 1677.10 | 1676.62 | 1694.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1693.10 | 1680.60 | 1690.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1741.70 | 1680.60 | 1690.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1732.90 | 1691.06 | 1694.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 1720.40 | 1691.06 | 1694.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1731.40 | 1699.13 | 1698.00 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1700.00 | 1705.43 | 1705.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1681.90 | 1699.01 | 1702.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1651.50 | 1620.76 | 1637.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1651.50 | 1620.76 | 1637.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1630.80 | 1622.76 | 1637.24 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 1652.40 | 1642.25 | 1641.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1695.70 | 1656.38 | 1648.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1909.00 | 1921.37 | 1879.18 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1824.00 | 1865.58 | 1868.58 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 09:15:00 | 1901.70 | 1872.80 | 1871.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 1938.00 | 1890.92 | 1881.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1969.00 | 1976.12 | 1954.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:30:00 | 1990.20 | 1978.95 | 1966.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 1990.30 | 1981.55 | 1969.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:30:00 | 1996.90 | 1984.24 | 1971.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1999.30 | 1986.03 | 1976.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1999.00 | 1988.63 | 1978.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 2023.40 | 1997.59 | 1987.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 2019.90 | 2057.82 | 2058.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 2019.90 | 2057.82 | 2058.60 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 2070.10 | 2054.53 | 2052.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 2081.80 | 2064.89 | 2058.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 2119.20 | 2123.31 | 2099.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 2103.50 | 2124.09 | 2109.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 2103.50 | 2124.09 | 2109.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 2103.50 | 2124.09 | 2109.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 2090.00 | 2117.27 | 2107.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 2086.20 | 2117.27 | 2107.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 2106.40 | 2112.37 | 2106.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:15:00 | 2112.30 | 2109.49 | 2106.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:45:00 | 2113.20 | 2110.26 | 2107.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 2142.70 | 2109.41 | 2107.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 2152.10 | 2163.43 | 2164.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 2152.10 | 2163.43 | 2164.59 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 09:15:00 | 2173.70 | 2165.48 | 2165.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 10:15:00 | 2186.00 | 2169.59 | 2167.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 2151.00 | 2166.82 | 2166.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:45:00 | 2146.50 | 2166.82 | 2166.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 2174.00 | 2168.25 | 2167.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 2188.40 | 2173.46 | 2169.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2108.00 | 2164.46 | 2166.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2108.00 | 2164.46 | 2166.46 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 2220.00 | 2162.43 | 2156.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 2235.10 | 2176.97 | 2163.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2162.40 | 2228.63 | 2212.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:30:00 | 2133.70 | 2228.63 | 2212.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2142.10 | 2211.33 | 2206.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:45:00 | 2149.00 | 2211.33 | 2206.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 2143.40 | 2197.74 | 2200.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 2133.70 | 2175.69 | 2189.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2256.20 | 2187.00 | 2190.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 2256.20 | 2187.00 | 2190.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 2245.30 | 2198.66 | 2195.56 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 2193.30 | 2204.84 | 2205.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2167.60 | 2194.22 | 2200.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2179.50 | 2176.70 | 2188.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 2189.80 | 2176.70 | 2188.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 2186.80 | 2178.72 | 2188.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 2179.30 | 2178.72 | 2188.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 2180.00 | 2178.97 | 2187.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 2154.00 | 2178.97 | 2187.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 2305.80 | 2164.88 | 2151.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 2305.80 | 2164.88 | 2151.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 09:15:00 | 2316.00 | 2267.80 | 2254.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 2323.50 | 2338.51 | 2304.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 2323.50 | 2338.51 | 2304.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 2296.30 | 2330.07 | 2303.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 2311.80 | 2330.07 | 2303.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 2281.20 | 2320.30 | 2301.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:30:00 | 2281.50 | 2320.30 | 2301.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 2270.10 | 2301.87 | 2295.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:30:00 | 2271.60 | 2301.87 | 2295.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 2269.00 | 2290.81 | 2291.54 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 2450.80 | 2322.81 | 2306.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2543.50 | 2429.46 | 2374.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 2512.20 | 2516.66 | 2470.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:30:00 | 2511.00 | 2516.66 | 2470.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2554.90 | 2532.35 | 2507.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:15:00 | 2572.00 | 2532.35 | 2507.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 2459.20 | 2524.06 | 2508.06 | SL hit (close<static) qty=1.00 sl=2496.10 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 2390.40 | 2497.33 | 2497.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 11:15:00 | 2276.40 | 2453.14 | 2477.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 2166.60 | 2154.26 | 2245.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:45:00 | 2168.20 | 2154.26 | 2245.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 2245.00 | 2179.73 | 2241.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 2245.00 | 2179.73 | 2241.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2210.90 | 2185.96 | 2238.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 15:15:00 | 2205.00 | 2185.96 | 2238.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2285.00 | 2231.70 | 2236.43 | SL hit (close>static) qty=1.00 sl=2247.70 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 2287.80 | 2246.17 | 2242.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 2296.60 | 2256.26 | 2247.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 2321.20 | 2327.93 | 2295.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 2321.20 | 2327.93 | 2295.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2333.50 | 2354.51 | 2334.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 2348.00 | 2346.17 | 2333.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 12:15:00 | 2307.40 | 2337.17 | 2339.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 12:15:00 | 2307.40 | 2337.17 | 2339.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 13:15:00 | 2286.80 | 2327.10 | 2334.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 2325.40 | 2316.55 | 2325.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 2323.00 | 2316.55 | 2325.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 2309.00 | 2315.04 | 2324.28 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2374.00 | 2336.30 | 2331.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 2392.60 | 2369.84 | 2355.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 2397.60 | 2405.01 | 2385.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 2391.10 | 2405.01 | 2385.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 2388.00 | 2401.61 | 2385.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 2388.00 | 2401.61 | 2385.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2376.40 | 2396.56 | 2384.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:00:00 | 2376.40 | 2396.56 | 2384.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2362.50 | 2389.75 | 2382.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 2378.90 | 2389.75 | 2382.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2409.70 | 2418.42 | 2406.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 2421.50 | 2418.42 | 2406.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2397.10 | 2414.16 | 2405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 2399.60 | 2414.16 | 2405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 2396.30 | 2410.59 | 2404.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 2396.60 | 2410.59 | 2404.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 2381.70 | 2398.79 | 2400.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 2371.90 | 2390.73 | 2396.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 2397.60 | 2384.65 | 2391.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 2408.00 | 2384.65 | 2391.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 2406.00 | 2388.92 | 2392.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 2406.00 | 2388.92 | 2392.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 2394.80 | 2391.71 | 2393.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 2424.00 | 2391.71 | 2393.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 2436.00 | 2400.57 | 2397.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 2454.00 | 2411.26 | 2402.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 2482.00 | 2487.53 | 2457.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 13:00:00 | 2482.00 | 2487.53 | 2457.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 2470.00 | 2478.96 | 2460.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 2512.70 | 2478.96 | 2460.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2490.50 | 2481.27 | 2463.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 2558.60 | 2495.52 | 2479.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 2575.40 | 2496.75 | 2484.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-02 10:15:00 | 1157.00 | 2025-06-06 10:15:00 | 1155.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-06-02 11:00:00 | 1160.70 | 2025-06-06 10:15:00 | 1155.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1118.80 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-06-11 12:00:00 | 1116.10 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1119.20 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-06-12 13:30:00 | 1118.40 | 2025-06-18 14:15:00 | 1108.50 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2025-06-25 12:30:00 | 1092.80 | 2025-06-26 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-25 14:15:00 | 1092.70 | 2025-06-26 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-03 14:45:00 | 1099.00 | 2025-07-04 11:15:00 | 1120.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-07-04 09:15:00 | 1097.40 | 2025-07-04 11:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-11 13:45:00 | 1144.60 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 3.85% |
| BUY | retest2 | 2025-07-11 14:15:00 | 1147.20 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2025-07-14 10:15:00 | 1159.00 | 2025-07-21 13:15:00 | 1188.70 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2025-07-23 15:00:00 | 1185.30 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-24 09:15:00 | 1186.10 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1179.80 | 2025-07-30 09:15:00 | 1190.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-31 09:15:00 | 1247.50 | 2025-08-01 09:15:00 | 1372.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-11 09:15:00 | 1270.80 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-08-11 11:30:00 | 1271.10 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1271.40 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-08-11 14:15:00 | 1273.20 | 2025-08-12 11:15:00 | 1324.60 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-08-26 13:45:00 | 1422.60 | 2025-08-28 14:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-28 11:00:00 | 1423.20 | 2025-08-28 14:15:00 | 1401.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1453.50 | 2025-09-26 09:15:00 | 1380.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 1453.50 | 2025-09-26 09:15:00 | 1380.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1449.10 | 2025-09-26 09:15:00 | 1376.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1453.50 | 2025-09-30 14:15:00 | 1308.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 1453.50 | 2025-09-30 14:15:00 | 1308.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 1449.10 | 2025-10-01 09:15:00 | 1304.19 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-10-07 11:00:00 | 1406.00 | 2025-10-15 10:15:00 | 1546.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1425.00 | 2025-10-15 10:15:00 | 1547.70 | TARGET_HIT | 1.00 | 8.61% |
| BUY | retest2 | 2025-10-08 10:45:00 | 1407.00 | 2025-10-15 10:15:00 | 1548.80 | TARGET_HIT | 1.00 | 10.08% |
| BUY | retest2 | 2025-10-09 11:15:00 | 1408.00 | 2025-10-16 12:15:00 | 1567.50 | TARGET_HIT | 1.00 | 11.33% |
| BUY | retest2 | 2025-10-14 13:45:00 | 1488.40 | 2025-10-17 12:15:00 | 1637.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:30:00 | 1494.40 | 2025-10-17 12:15:00 | 1643.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1705.10 | 2025-11-19 09:15:00 | 1720.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-20 12:45:00 | 1727.10 | 2025-11-21 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-21 13:00:00 | 1723.00 | 2025-11-21 14:15:00 | 1697.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-21 14:00:00 | 1719.20 | 2025-11-21 14:15:00 | 1697.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-27 12:30:00 | 1792.60 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-27 15:00:00 | 1792.10 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1792.10 | 2025-12-01 14:15:00 | 1769.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2025-12-08 09:15:00 | 1693.50 | 2025-12-10 09:15:00 | 1701.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-09 14:00:00 | 1675.40 | 2025-12-10 09:15:00 | 1701.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-12 11:45:00 | 1654.00 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-12 13:15:00 | 1656.40 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1648.60 | 2025-12-16 11:15:00 | 1675.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1671.30 | 2025-12-30 12:15:00 | 1652.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-30 10:00:00 | 1663.50 | 2025-12-30 12:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-06 14:15:00 | 1770.90 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1773.00 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-01-07 11:15:00 | 1770.00 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-07 12:30:00 | 1776.80 | 2026-01-08 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-02-06 09:30:00 | 1990.20 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2026-02-06 12:00:00 | 1990.30 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2026-02-06 12:30:00 | 1996.90 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1999.30 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2026-02-10 09:15:00 | 2023.40 | 2026-02-13 14:15:00 | 2019.90 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-02-20 13:15:00 | 2112.30 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2026-02-20 14:45:00 | 2113.20 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2026-02-23 09:15:00 | 2142.70 | 2026-02-27 15:15:00 | 2152.10 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2026-03-02 14:45:00 | 2188.40 | 2026-03-04 09:15:00 | 2108.00 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-03-13 09:15:00 | 2154.00 | 2026-03-17 09:15:00 | 2305.80 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest2 | 2026-03-30 15:15:00 | 2572.00 | 2026-04-01 09:15:00 | 2459.20 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2026-04-06 15:15:00 | 2205.00 | 2026-04-08 09:15:00 | 2285.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2026-04-13 12:00:00 | 2348.00 | 2026-04-15 12:15:00 | 2307.40 | STOP_HIT | 1.00 | -1.73% |
