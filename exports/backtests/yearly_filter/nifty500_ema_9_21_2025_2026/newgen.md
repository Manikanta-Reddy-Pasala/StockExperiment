# Newgen Software Technologies Ltd. (NEWGEN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 506.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 44 |
| ALERT2 | 40 |
| ALERT2_SKIP | 27 |
| ALERT3 | 95 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 25
- **Target hits / Stop hits / Partials:** 2 / 35 / 8
- **Avg / median % per leg:** 0.27% / -0.38%
- **Sum % (uncompounded):** 12.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 3 | 50.0% | 2 | 4 | 0 | 3.26% | 19.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 3 | 50.0% | 2 | 4 | 0 | 3.26% | 19.5% |
| SELL (all) | 39 | 17 | 43.6% | 0 | 31 | 8 | -0.19% | -7.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 17 | 43.6% | 0 | 31 | 8 | -0.19% | -7.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 20 | 44.4% | 2 | 35 | 8 | 0.27% | 12.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 1250.00 | 1254.99 | 1255.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1247.25 | 1250.47 | 1252.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1248.20 | 1225.61 | 1234.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1250.40 | 1225.61 | 1234.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1248.75 | 1230.24 | 1235.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1248.70 | 1230.24 | 1235.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 1247.90 | 1238.55 | 1238.32 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 1233.95 | 1237.63 | 1237.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1225.60 | 1234.72 | 1236.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1231.00 | 1229.56 | 1232.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1231.00 | 1229.56 | 1232.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1231.20 | 1229.89 | 1232.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1234.10 | 1229.89 | 1232.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1226.30 | 1229.17 | 1232.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 1218.90 | 1226.80 | 1230.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 12:45:00 | 1222.30 | 1225.14 | 1226.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1269.00 | 1233.49 | 1230.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1269.00 | 1233.49 | 1230.06 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 1220.00 | 1230.15 | 1230.94 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1234.90 | 1231.53 | 1231.19 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1226.00 | 1230.00 | 1230.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 1222.00 | 1227.35 | 1228.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1174.20 | 1169.23 | 1182.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 1174.20 | 1169.23 | 1182.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1240.10 | 1183.40 | 1187.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1240.10 | 1183.40 | 1187.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1227.00 | 1192.12 | 1190.90 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 1186.00 | 1206.56 | 1208.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 13:15:00 | 1179.50 | 1190.15 | 1196.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1147.90 | 1133.89 | 1148.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 1147.90 | 1133.89 | 1148.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1145.20 | 1136.15 | 1148.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 1145.00 | 1136.15 | 1148.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1151.90 | 1139.30 | 1148.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 1151.90 | 1139.30 | 1148.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1150.60 | 1141.56 | 1148.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 1149.80 | 1141.56 | 1148.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 1148.80 | 1143.01 | 1148.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 1150.60 | 1143.01 | 1148.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 1150.80 | 1144.57 | 1148.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 1154.90 | 1144.57 | 1148.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1154.90 | 1146.63 | 1149.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1189.50 | 1146.63 | 1149.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 1224.60 | 1162.23 | 1156.19 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 1144.20 | 1166.16 | 1168.62 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 1170.30 | 1167.86 | 1167.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 15:15:00 | 1175.00 | 1169.29 | 1168.51 | Break + close above crossover candle high |

### Cycle 13 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1160.00 | 1167.43 | 1167.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 1153.00 | 1162.36 | 1165.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1135.00 | 1131.71 | 1137.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 1135.70 | 1131.71 | 1137.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1134.00 | 1123.32 | 1128.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1134.00 | 1123.32 | 1128.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1132.90 | 1125.24 | 1128.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1117.10 | 1129.09 | 1129.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 1090.90 | 1088.76 | 1088.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1090.90 | 1088.76 | 1088.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1094.00 | 1089.81 | 1089.00 | Break + close above crossover candle high |

### Cycle 15 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1027.90 | 1079.07 | 1084.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 972.70 | 1036.67 | 1060.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 866.90 | 860.29 | 873.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 869.00 | 860.29 | 873.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 857.70 | 854.25 | 860.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 863.80 | 854.25 | 860.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 859.40 | 855.28 | 860.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 864.90 | 855.28 | 860.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 860.20 | 856.26 | 860.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 854.00 | 856.91 | 859.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 853.50 | 856.23 | 859.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 855.50 | 852.79 | 855.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 909.05 | 849.63 | 843.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 909.05 | 849.63 | 843.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 12:15:00 | 934.00 | 876.96 | 857.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 15:15:00 | 917.00 | 929.74 | 906.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:15:00 | 910.25 | 929.74 | 906.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 904.40 | 924.67 | 906.75 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 887.75 | 901.65 | 902.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 885.50 | 895.99 | 899.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 872.10 | 871.43 | 880.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:45:00 | 863.25 | 869.13 | 876.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 858.00 | 868.28 | 875.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 857.30 | 864.55 | 872.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 917.25 | 869.40 | 869.44 | SL hit (close>static) qty=1.00 sl=886.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 919.20 | 879.36 | 873.96 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 875.75 | 882.34 | 882.55 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 895.45 | 881.54 | 881.22 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 877.25 | 881.01 | 881.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 871.10 | 876.94 | 879.13 | Break + close below crossover candle low |

### Cycle 22 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 914.75 | 879.40 | 877.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 933.45 | 890.21 | 882.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 899.10 | 906.87 | 896.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 895.10 | 906.87 | 896.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 899.35 | 905.36 | 896.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 897.00 | 905.36 | 896.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 904.40 | 905.17 | 897.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:45:00 | 897.70 | 905.17 | 897.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 897.85 | 903.71 | 897.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 897.85 | 903.71 | 897.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 898.65 | 902.70 | 897.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 905.00 | 899.48 | 896.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 887.05 | 897.05 | 896.92 | SL hit (close<static) qty=1.00 sl=896.50 alert=retest2 |

### Cycle 23 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 894.40 | 897.22 | 897.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 888.80 | 894.59 | 895.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 891.20 | 890.08 | 893.06 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 894.55 | 892.24 | 892.17 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 891.40 | 892.34 | 892.36 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 914.85 | 896.84 | 894.40 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 890.35 | 898.52 | 899.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 885.05 | 895.83 | 897.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 888.00 | 881.40 | 884.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 893.00 | 881.40 | 884.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 888.75 | 882.87 | 885.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 888.75 | 882.87 | 885.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 887.70 | 883.84 | 885.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:30:00 | 886.85 | 883.84 | 885.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 893.45 | 885.76 | 886.03 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 890.80 | 886.77 | 886.46 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 883.45 | 886.10 | 886.19 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 889.00 | 886.73 | 886.46 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 885.25 | 886.67 | 886.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 884.55 | 886.12 | 886.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 887.00 | 886.30 | 886.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 887.00 | 886.30 | 886.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 885.30 | 886.10 | 886.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:30:00 | 884.00 | 885.83 | 886.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:00:00 | 883.85 | 885.44 | 885.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 883.50 | 885.58 | 885.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 884.25 | 885.46 | 885.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 884.85 | 885.34 | 885.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 885.50 | 885.34 | 885.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 884.30 | 885.13 | 885.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 893.25 | 885.13 | 885.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 886.25 | 885.36 | 885.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 887.25 | 885.90 | 885.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 887.25 | 885.90 | 885.85 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 884.85 | 885.69 | 885.76 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 932.65 | 894.94 | 889.93 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 898.60 | 907.16 | 907.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 891.10 | 899.65 | 903.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 917.50 | 886.96 | 889.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:30:00 | 917.00 | 886.96 | 889.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 917.60 | 893.08 | 891.79 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 890.00 | 894.62 | 895.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 887.80 | 893.26 | 894.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 888.35 | 880.84 | 883.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 888.35 | 880.84 | 883.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 887.60 | 882.19 | 883.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 910.00 | 882.19 | 883.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 903.00 | 886.35 | 885.69 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 883.15 | 890.83 | 891.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 879.00 | 882.64 | 884.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 15:15:00 | 872.00 | 871.85 | 875.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 876.95 | 871.85 | 875.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 877.35 | 872.95 | 875.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 872.10 | 873.26 | 875.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 872.50 | 873.26 | 875.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 868.05 | 872.70 | 874.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 903.15 | 864.88 | 865.55 | SL hit (close>static) qty=1.00 sl=881.90 alert=retest2 |

### Cycle 40 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 899.60 | 871.83 | 868.64 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 873.95 | 876.23 | 876.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 870.85 | 874.90 | 875.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 878.45 | 875.23 | 875.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 878.85 | 875.23 | 875.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 879.55 | 876.09 | 876.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 878.20 | 876.09 | 876.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 879.80 | 876.83 | 876.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 883.25 | 878.12 | 877.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 891.80 | 898.72 | 891.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 890.60 | 898.72 | 891.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 891.80 | 897.33 | 891.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 904.00 | 894.68 | 893.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-28 14:15:00 | 994.40 | 929.31 | 911.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 970.10 | 977.54 | 977.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 957.00 | 972.47 | 975.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 961.50 | 951.21 | 959.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 961.50 | 951.21 | 959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 947.80 | 950.53 | 958.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 944.90 | 950.42 | 957.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:45:00 | 943.40 | 943.47 | 950.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 963.30 | 953.32 | 952.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 963.30 | 953.32 | 952.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 971.00 | 959.10 | 955.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 977.35 | 977.58 | 968.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:45:00 | 980.35 | 977.58 | 968.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 972.60 | 975.50 | 969.40 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 941.15 | 963.64 | 965.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 929.90 | 944.56 | 953.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 929.60 | 923.01 | 930.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:45:00 | 929.50 | 923.01 | 930.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 927.35 | 923.88 | 929.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 926.10 | 925.10 | 929.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 918.35 | 924.90 | 928.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 09:15:00 | 879.79 | 890.90 | 900.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 872.43 | 883.48 | 894.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 881.00 | 879.42 | 888.60 | SL hit (close>ema200) qty=0.50 sl=879.42 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 879.90 | 877.90 | 877.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 891.60 | 880.64 | 879.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 890.50 | 892.66 | 887.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:30:00 | 899.90 | 892.66 | 887.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 887.35 | 891.60 | 887.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 887.35 | 891.60 | 887.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 883.75 | 890.03 | 887.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 883.75 | 890.03 | 887.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 884.85 | 888.99 | 887.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 881.25 | 888.99 | 887.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 879.95 | 887.19 | 886.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 878.05 | 887.19 | 886.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 879.00 | 885.55 | 885.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 866.95 | 881.83 | 884.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 847.45 | 846.41 | 857.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:45:00 | 841.40 | 846.41 | 857.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 855.20 | 848.26 | 856.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 857.75 | 848.26 | 856.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 847.30 | 848.07 | 855.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 856.65 | 848.07 | 855.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 847.80 | 840.17 | 846.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 847.80 | 840.17 | 846.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 846.85 | 841.51 | 846.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 845.10 | 843.97 | 846.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 856.55 | 847.61 | 848.12 | SL hit (close>static) qty=1.00 sl=853.60 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 853.35 | 848.76 | 848.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 862.60 | 857.70 | 854.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 856.50 | 858.90 | 856.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 856.50 | 858.90 | 856.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 854.15 | 857.95 | 855.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 854.15 | 857.95 | 855.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 855.00 | 857.36 | 855.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:30:00 | 854.30 | 857.36 | 855.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 851.95 | 856.28 | 855.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 851.95 | 856.28 | 855.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 849.05 | 854.83 | 854.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 844.90 | 852.85 | 853.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 839.95 | 838.75 | 843.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 839.95 | 838.75 | 843.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 836.65 | 837.17 | 840.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 831.00 | 834.66 | 839.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 871.60 | 837.22 | 837.38 | SL hit (close>static) qty=1.00 sl=844.40 alert=retest2 |

### Cycle 50 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 856.50 | 841.08 | 839.12 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 845.35 | 848.86 | 849.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 843.20 | 847.27 | 848.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 835.00 | 831.28 | 836.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 835.00 | 831.28 | 836.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 826.00 | 830.22 | 835.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 835.00 | 830.22 | 835.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 836.10 | 831.62 | 835.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 836.10 | 831.62 | 835.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 841.75 | 833.65 | 835.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 842.70 | 833.65 | 835.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 845.15 | 837.35 | 837.09 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 833.10 | 837.85 | 838.33 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 843.20 | 838.23 | 838.07 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 835.80 | 837.67 | 837.83 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 842.00 | 838.54 | 838.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 12:15:00 | 843.20 | 839.47 | 838.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 834.80 | 838.53 | 838.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 834.80 | 838.53 | 838.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 833.00 | 837.43 | 837.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 15:15:00 | 832.05 | 836.35 | 837.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 841.15 | 829.32 | 831.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 841.15 | 829.32 | 831.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 837.85 | 831.02 | 832.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 833.75 | 833.06 | 833.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 838.60 | 834.16 | 833.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 838.60 | 834.16 | 833.63 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 828.20 | 833.14 | 833.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 825.05 | 829.50 | 831.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 780.60 | 779.47 | 790.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 781.20 | 779.47 | 790.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 587.80 | 583.50 | 588.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 587.80 | 583.50 | 588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 579.20 | 582.64 | 587.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 571.05 | 579.99 | 585.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-05 13:15:00 | 542.50 | 550.01 | 556.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 554.95 | 538.46 | 544.04 | SL hit (close>ema200) qty=0.50 sl=538.46 alert=retest2 |

### Cycle 60 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 565.15 | 547.95 | 547.64 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 539.90 | 550.50 | 551.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 11:15:00 | 537.65 | 547.93 | 550.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 501.45 | 475.89 | 487.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 501.45 | 475.89 | 487.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 519.00 | 484.51 | 489.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 519.00 | 484.51 | 489.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 528.30 | 500.39 | 496.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 593.90 | 535.46 | 519.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 582.95 | 592.78 | 563.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:45:00 | 581.30 | 592.78 | 563.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 561.35 | 577.46 | 566.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 561.35 | 577.46 | 566.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 550.00 | 571.97 | 564.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 546.15 | 571.97 | 564.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 539.20 | 560.75 | 560.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 539.60 | 560.75 | 560.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 539.95 | 556.59 | 558.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 12:15:00 | 533.80 | 552.03 | 556.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 517.50 | 507.90 | 518.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 528.50 | 507.90 | 518.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 515.35 | 509.39 | 518.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:00:00 | 511.20 | 510.41 | 517.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 509.65 | 512.12 | 515.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 485.64 | 507.71 | 511.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 484.17 | 507.71 | 511.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 494.80 | 493.87 | 501.11 | SL hit (close>ema200) qty=0.50 sl=493.87 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 461.70 | 448.31 | 448.01 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 444.70 | 451.68 | 451.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 441.10 | 448.38 | 450.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 446.55 | 445.77 | 448.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 447.90 | 445.77 | 448.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 445.30 | 445.68 | 447.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 443.80 | 445.68 | 447.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 443.70 | 445.38 | 447.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 444.50 | 444.98 | 447.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 421.61 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 421.51 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 422.27 | 432.49 | 439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 429.50 | 426.57 | 432.74 | SL hit (close>ema200) qty=0.50 sl=426.57 alert=retest2 |

### Cycle 66 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 443.80 | 436.28 | 435.40 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 429.35 | 435.00 | 435.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 420.00 | 431.52 | 433.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 431.85 | 414.94 | 420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 431.85 | 414.94 | 420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 427.70 | 417.49 | 421.19 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 430.85 | 424.29 | 423.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 432.35 | 426.91 | 425.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 418.85 | 425.29 | 424.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 427.40 | 424.93 | 424.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 15:15:00 | 450.00 | 453.73 | 453.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 15:15:00 | 450.00 | 453.73 | 453.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 436.90 | 450.37 | 452.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 14:15:00 | 446.40 | 445.97 | 448.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 446.40 | 445.97 | 448.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 70 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 470.90 | 450.80 | 450.62 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 468.45 | 475.08 | 475.71 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 480.50 | 475.18 | 475.07 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 469.60 | 475.79 | 476.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 462.65 | 469.08 | 472.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 486.25 | 470.64 | 472.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 489.05 | 470.64 | 472.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 491.00 | 474.71 | 473.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 504.00 | 480.57 | 476.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 487.35 | 488.51 | 483.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 487.35 | 488.51 | 483.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 483.20 | 487.72 | 484.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 483.20 | 487.72 | 484.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 483.00 | 486.78 | 484.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 483.00 | 486.78 | 484.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 483.80 | 486.18 | 484.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 490.60 | 486.18 | 484.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 478.10 | 484.15 | 484.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 478.10 | 484.15 | 484.37 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 493.25 | 484.68 | 484.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 515.55 | 490.85 | 487.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 10:15:00 | 497.50 | 498.84 | 493.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 10:45:00 | 495.00 | 498.84 | 493.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 494.40 | 497.43 | 493.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:15:00 | 492.80 | 497.43 | 493.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 496.45 | 497.23 | 493.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 495.40 | 497.23 | 493.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 494.95 | 496.78 | 494.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 494.95 | 496.78 | 494.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 489.40 | 495.34 | 493.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 489.40 | 495.34 | 493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 487.40 | 493.75 | 493.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 487.75 | 493.75 | 493.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 488.30 | 492.66 | 492.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 485.15 | 490.28 | 491.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 491.50 | 490.15 | 491.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 496.00 | 490.15 | 491.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 487.10 | 489.54 | 490.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 484.50 | 488.07 | 489.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 484.80 | 487.48 | 489.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 500.25 | 492.36 | 491.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 500.25 | 492.36 | 491.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 506.70 | 499.69 | 495.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 1144.90 | 2025-05-20 10:15:00 | 1259.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-03 12:00:00 | 1218.90 | 2025-06-05 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-06-04 12:45:00 | 1222.30 | 2025-06-05 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-07-10 09:15:00 | 1117.10 | 2025-07-16 15:15:00 | 1090.90 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-07-30 13:45:00 | 854.00 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.45% |
| SELL | retest2 | 2025-07-30 15:00:00 | 853.50 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest2 | 2025-07-31 13:45:00 | 855.50 | 2025-08-05 10:15:00 | 909.05 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2025-08-12 12:45:00 | 863.25 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2025-08-12 15:15:00 | 858.00 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.91% |
| SELL | retest2 | 2025-08-13 09:30:00 | 857.30 | 2025-08-14 09:15:00 | 917.25 | STOP_HIT | 1.00 | -6.99% |
| BUY | retest2 | 2025-08-28 09:15:00 | 905.00 | 2025-08-28 13:15:00 | 887.05 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-28 15:00:00 | 902.50 | 2025-08-29 09:15:00 | 891.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-15 13:30:00 | 884.00 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-15 15:00:00 | 883.85 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-09-16 12:45:00 | 883.50 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-16 13:45:00 | 884.25 | 2025-09-17 12:15:00 | 887.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-10 12:30:00 | 872.10 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-10-10 13:15:00 | 872.50 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-10-13 09:15:00 | 868.05 | 2025-10-15 09:15:00 | 903.15 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2025-10-28 09:15:00 | 904.00 | 2025-10-28 14:15:00 | 994.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-10 09:15:00 | 944.90 | 2025-11-11 12:15:00 | 963.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-11-10 14:45:00 | 943.40 | 2025-11-11 12:15:00 | 963.30 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-11-19 11:45:00 | 926.10 | 2025-11-25 09:15:00 | 879.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:00:00 | 918.35 | 2025-11-25 12:15:00 | 872.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 11:45:00 | 926.10 | 2025-11-26 09:15:00 | 881.00 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-11-19 15:00:00 | 918.35 | 2025-11-26 09:15:00 | 881.00 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-12-11 14:30:00 | 845.10 | 2025-12-12 09:15:00 | 856.55 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-19 10:30:00 | 831.00 | 2025-12-22 09:15:00 | 871.60 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-01-07 13:45:00 | 833.75 | 2026-01-07 14:15:00 | 838.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-02-01 15:15:00 | 571.05 | 2026-02-05 13:15:00 | 542.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 15:15:00 | 571.05 | 2026-02-09 09:15:00 | 554.95 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-02-26 13:00:00 | 511.20 | 2026-03-02 09:15:00 | 485.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 10:45:00 | 509.65 | 2026-03-02 09:15:00 | 484.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:00:00 | 511.20 | 2026-03-04 09:15:00 | 494.80 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2026-02-27 10:45:00 | 509.65 | 2026-03-04 09:15:00 | 494.80 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-03-20 12:15:00 | 443.80 | 2026-03-23 12:15:00 | 421.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 443.70 | 2026-03-23 12:15:00 | 421.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:30:00 | 444.50 | 2026-03-23 12:15:00 | 422.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 443.80 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2026-03-20 12:45:00 | 443.70 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-03-20 13:30:00 | 444.50 | 2026-03-24 11:15:00 | 429.50 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2026-04-02 11:45:00 | 427.40 | 2026-04-10 15:15:00 | 450.00 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2026-04-29 09:15:00 | 490.60 | 2026-04-29 15:15:00 | 478.10 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-05-06 12:30:00 | 484.50 | 2026-05-07 10:15:00 | 500.25 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-05-06 13:45:00 | 484.80 | 2026-05-07 10:15:00 | 500.25 | STOP_HIT | 1.00 | -3.19% |
