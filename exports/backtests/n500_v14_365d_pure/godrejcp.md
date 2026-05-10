# Godrej Consumer Products Ltd. (GODREJCP)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1041.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 8
- **Winners / losers:** 7 / 17
- **Target hits / Stop hits / Partials:** 0 / 17 / 7
- **Avg / median % per leg:** -0.36% / -0.72%
- **Sum % (uncompounded):** -8.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.88% | -13.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.88% | -13.2% |
| SELL (all) | 17 | 7 | 41.2% | 0 | 10 | 7 | 0.27% | 4.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 7 | 41.2% | 0 | 10 | 7 | 0.27% | 4.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 7 | 29.2% | 0 | 17 | 7 | -0.36% | -8.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1171.40 | 1206.12 | 1206.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 1165.80 | 1204.26 | 1205.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1246.00 | 1199.77 | 1202.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1246.00 | 1199.77 | 1202.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1246.00 | 1199.77 | 1202.71 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1274.40 | 1206.18 | 1205.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1277.30 | 1207.54 | 1206.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1236.60 | 1237.08 | 1224.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:30:00 | 1231.90 | 1237.08 | 1224.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1227.00 | 1236.99 | 1225.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 1225.30 | 1236.99 | 1225.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1228.80 | 1236.91 | 1225.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 1223.50 | 1236.91 | 1225.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 1226.60 | 1236.81 | 1225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:30:00 | 1224.60 | 1236.81 | 1225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1225.60 | 1236.70 | 1225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1225.00 | 1236.70 | 1225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1223.50 | 1236.57 | 1225.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1224.20 | 1236.57 | 1225.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1223.00 | 1236.43 | 1225.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1221.40 | 1236.43 | 1225.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 1224.50 | 1234.90 | 1225.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 1224.50 | 1234.90 | 1225.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1214.50 | 1234.70 | 1224.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 1214.50 | 1234.70 | 1224.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1210.10 | 1234.45 | 1224.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 14:00:00 | 1210.10 | 1234.45 | 1224.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1240.40 | 1235.28 | 1226.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 1227.00 | 1235.28 | 1226.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1226.40 | 1235.71 | 1227.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:30:00 | 1221.50 | 1235.71 | 1227.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1229.40 | 1235.65 | 1227.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 1229.40 | 1235.65 | 1227.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1220.30 | 1235.50 | 1227.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1220.30 | 1235.50 | 1227.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1219.50 | 1235.34 | 1227.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 1219.50 | 1235.34 | 1227.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1206.70 | 1230.79 | 1225.40 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1184.80 | 1220.60 | 1220.71 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 1245.10 | 1220.93 | 1220.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 12:15:00 | 1249.90 | 1221.81 | 1221.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1223.40 | 1240.27 | 1232.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 1223.40 | 1240.27 | 1232.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1223.40 | 1240.27 | 1232.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1223.40 | 1240.27 | 1232.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1223.90 | 1240.10 | 1232.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1223.90 | 1240.10 | 1232.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1234.00 | 1239.97 | 1232.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 1238.70 | 1239.79 | 1232.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1229.50 | 1239.44 | 1232.18 | SL hit (close<static) qty=1.00 sl=1231.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:00:00 | 1237.00 | 1239.27 | 1232.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 1237.50 | 1239.28 | 1232.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 1230.00 | 1242.51 | 1235.23 | SL hit (close<static) qty=1.00 sl=1231.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 1230.00 | 1242.51 | 1235.23 | SL hit (close<static) qty=1.00 sl=1231.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1238.40 | 1242.00 | 1235.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1235.90 | 1241.94 | 1235.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:45:00 | 1237.00 | 1241.94 | 1235.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1241.30 | 1241.93 | 1235.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1236.00 | 1241.93 | 1235.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1235.40 | 1241.87 | 1235.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 1237.80 | 1241.87 | 1235.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1237.00 | 1241.73 | 1235.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1229.80 | 1241.61 | 1235.23 | SL hit (close<static) qty=1.00 sl=1231.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1239.90 | 1241.04 | 1235.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 1241.50 | 1240.97 | 1235.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1232.60 | 1240.93 | 1235.30 | SL hit (close<static) qty=1.00 sl=1233.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1232.60 | 1240.93 | 1235.30 | SL hit (close<static) qty=1.00 sl=1233.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 1174.20 | 1230.20 | 1230.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 1162.80 | 1227.89 | 1229.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1171.60 | 1147.17 | 1174.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1171.60 | 1147.17 | 1174.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1171.60 | 1147.17 | 1174.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1139.90 | 1149.25 | 1173.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1126.20 | 1148.87 | 1172.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1133.10 | 1145.27 | 1168.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 1141.00 | 1145.02 | 1167.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1148.60 | 1140.00 | 1158.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 1145.60 | 1140.06 | 1158.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 1146.40 | 1140.06 | 1158.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:30:00 | 1145.20 | 1140.28 | 1158.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 1146.20 | 1140.84 | 1158.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1150.80 | 1141.08 | 1157.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1150.80 | 1141.08 | 1157.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1151.00 | 1136.50 | 1150.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1142.00 | 1136.50 | 1150.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1140.50 | 1136.54 | 1150.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1139.50 | 1136.54 | 1150.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1139.60 | 1136.64 | 1150.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 1160.00 | 1137.65 | 1150.17 | SL hit (close>static) qty=1.00 sl=1157.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 1160.00 | 1137.65 | 1150.17 | SL hit (close>static) qty=1.00 sl=1157.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1169.40 | 1138.87 | 1150.48 | SL hit (close>static) qty=1.00 sl=1167.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1169.40 | 1138.87 | 1150.48 | SL hit (close>static) qty=1.00 sl=1167.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1169.40 | 1138.87 | 1150.48 | SL hit (close>static) qty=1.00 sl=1167.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1169.40 | 1138.87 | 1150.48 | SL hit (close>static) qty=1.00 sl=1167.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1188.10 | 1139.36 | 1150.67 | SL hit (close>static) qty=1.00 sl=1184.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1188.10 | 1139.36 | 1150.67 | SL hit (close>static) qty=1.00 sl=1184.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1188.10 | 1139.36 | 1150.67 | SL hit (close>static) qty=1.00 sl=1184.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1188.10 | 1139.36 | 1150.67 | SL hit (close>static) qty=1.00 sl=1184.80 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1200.90 | 1159.20 | 1159.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 1206.60 | 1161.77 | 1160.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 1178.30 | 1214.74 | 1195.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 1178.30 | 1214.74 | 1195.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1178.30 | 1214.74 | 1195.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:45:00 | 1230.50 | 1198.34 | 1192.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 15:15:00 | 1116.70 | 1188.40 | 1188.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1116.70 | 1188.40 | 1188.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1070.10 | 1187.22 | 1188.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1074.10 | 1067.29 | 1110.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 09:45:00 | 1078.65 | 1067.29 | 1110.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1102.80 | 1070.59 | 1103.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 1094.00 | 1086.03 | 1106.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:00:00 | 1096.50 | 1086.21 | 1106.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:00:00 | 1092.15 | 1086.39 | 1106.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 1095.80 | 1086.58 | 1106.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1103.80 | 1087.08 | 1105.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 1087.00 | 1087.50 | 1105.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:30:00 | 1093.30 | 1085.79 | 1102.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 14:45:00 | 1094.40 | 1085.99 | 1102.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1045.00 | 1086.11 | 1102.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1039.30 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1041.67 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1037.54 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1041.01 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1038.63 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:15:00 | 1039.68 | 1085.70 | 1101.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:15:00 | 1032.65 | 1082.61 | 1099.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-08 09:30:00 | 1238.70 | 2025-09-08 14:15:00 | 1229.50 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-09 11:00:00 | 1237.00 | 2025-09-16 14:15:00 | 1230.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-09 11:30:00 | 1237.50 | 2025-09-16 14:15:00 | 1230.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-17 12:30:00 | 1238.40 | 2025-09-18 13:15:00 | 1229.80 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-19 15:15:00 | 1239.90 | 2025-09-22 13:15:00 | 1232.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-22 11:00:00 | 1241.50 | 2025-09-22 13:15:00 | 1232.60 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1139.90 | 2025-12-15 10:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1126.20 | 2025-12-15 10:15:00 | 1160.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-11-12 09:15:00 | 1133.10 | 2025-12-15 15:15:00 | 1169.40 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-11-12 12:00:00 | 1141.00 | 2025-12-15 15:15:00 | 1169.40 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-11-26 10:30:00 | 1145.60 | 2025-12-15 15:15:00 | 1169.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-11-26 11:00:00 | 1146.40 | 2025-12-15 15:15:00 | 1169.40 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-26 13:30:00 | 1145.20 | 2025-12-16 09:15:00 | 1188.10 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-11-27 11:45:00 | 1146.20 | 2025-12-16 09:15:00 | 1188.10 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1139.50 | 2025-12-16 09:15:00 | 1188.10 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1139.60 | 2025-12-16 09:15:00 | 1188.10 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest2 | 2026-02-24 14:45:00 | 1230.50 | 2026-03-06 15:15:00 | 1116.70 | STOP_HIT | 1.00 | -9.25% |
| SELL | retest2 | 2026-04-24 12:30:00 | 1094.00 | 2026-05-07 09:15:00 | 1039.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 10:00:00 | 1096.50 | 2026-05-07 09:15:00 | 1041.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 12:00:00 | 1092.15 | 2026-05-07 09:15:00 | 1037.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:30:00 | 1095.80 | 2026-05-07 09:15:00 | 1041.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 14:00:00 | 1087.00 | 2026-05-07 09:15:00 | 1038.63 | PARTIAL | 0.50 | 4.45% |
| SELL | retest2 | 2026-05-06 10:30:00 | 1093.30 | 2026-05-07 09:15:00 | 1039.68 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2026-05-06 14:45:00 | 1094.40 | 2026-05-08 09:15:00 | 1032.65 | PARTIAL | 0.50 | 5.64% |
