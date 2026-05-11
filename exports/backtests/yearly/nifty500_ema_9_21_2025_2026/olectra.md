# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1345.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 43 |
| ALERT2 | 41 |
| ALERT2_SKIP | 26 |
| ALERT3 | 99 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 11 |
| TARGET_HIT | 12 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 64 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 27
- **Target hits / Stop hits / Partials:** 12 / 41 / 11
- **Avg / median % per leg:** 2.44% / 2.88%
- **Sum % (uncompounded):** 156.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 13 | 54.2% | 10 | 14 | 0 | 3.30% | 79.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 13 | 54.2% | 10 | 14 | 0 | 3.30% | 79.2% |
| SELL (all) | 40 | 24 | 60.0% | 2 | 27 | 11 | 1.93% | 77.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 24 | 60.0% | 2 | 27 | 11 | 1.93% | 77.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 64 | 37 | 57.8% | 12 | 41 | 11 | 2.44% | 156.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1182.60 | 1139.94 | 1134.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1186.80 | 1149.31 | 1139.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1282.40 | 1284.74 | 1270.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:30:00 | 1280.00 | 1284.74 | 1270.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1270.00 | 1282.12 | 1272.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1268.80 | 1282.12 | 1272.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1289.10 | 1283.52 | 1274.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 1314.40 | 1283.52 | 1274.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 1263.00 | 1275.03 | 1273.51 | SL hit (close<static) qty=1.00 sl=1264.10 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1257.60 | 1271.55 | 1272.34 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1296.90 | 1273.99 | 1271.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1307.80 | 1283.33 | 1275.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1264.50 | 1290.54 | 1293.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 14:15:00 | 1255.20 | 1279.26 | 1287.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1254.70 | 1236.07 | 1235.09 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 1228.20 | 1236.86 | 1237.05 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 1240.00 | 1237.48 | 1237.32 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 1228.60 | 1235.87 | 1236.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1221.00 | 1230.12 | 1232.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1250.40 | 1229.29 | 1229.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 1237.00 | 1230.84 | 1230.22 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 1226.00 | 1230.03 | 1230.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 1218.00 | 1223.08 | 1226.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1225.10 | 1223.46 | 1225.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1223.00 | 1223.37 | 1225.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 1216.10 | 1222.64 | 1224.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1215.20 | 1220.80 | 1223.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1216.60 | 1219.06 | 1222.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 1155.29 | 1165.06 | 1172.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 1155.77 | 1165.06 | 1172.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1154.44 | 1161.77 | 1170.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 1156.30 | 1155.38 | 1163.36 | SL hit (close>ema200) qty=0.50 sl=1155.38 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1165.70 | 1157.78 | 1156.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1190.90 | 1165.63 | 1160.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1186.60 | 1188.61 | 1183.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1186.60 | 1188.61 | 1183.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1185.50 | 1187.98 | 1183.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 1185.30 | 1187.98 | 1183.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1182.30 | 1186.85 | 1183.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1182.30 | 1186.85 | 1183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1171.00 | 1183.68 | 1182.33 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1172.40 | 1181.42 | 1181.42 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 1215.00 | 1182.34 | 1180.46 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1201.60 | 1207.78 | 1208.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 1199.00 | 1206.03 | 1207.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 1210.10 | 1202.93 | 1202.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 1223.00 | 1206.94 | 1204.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 1206.50 | 1207.92 | 1205.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1207.60 | 1207.85 | 1205.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:15:00 | 1206.10 | 1207.85 | 1205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1205.70 | 1207.42 | 1205.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 1206.60 | 1207.42 | 1205.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1204.80 | 1206.90 | 1205.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1209.70 | 1206.72 | 1205.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 1217.10 | 1224.15 | 1224.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1217.10 | 1224.15 | 1224.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 14:15:00 | 1212.10 | 1219.10 | 1221.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1226.00 | 1223.15 | 1222.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1229.40 | 1224.26 | 1223.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1224.60 | 1225.27 | 1224.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1260.80 | 1232.37 | 1227.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1266.50 | 1243.19 | 1234.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 1279.40 | 1266.38 | 1258.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1268.00 | 1268.94 | 1264.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1267.00 | 1268.94 | 1264.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1267.00 | 1268.55 | 1264.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1283.30 | 1268.55 | 1264.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1272.40 | 1269.93 | 1266.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 1271.50 | 1270.24 | 1266.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1273.00 | 1270.49 | 1267.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-24 09:15:00 | 1393.15 | 1315.68 | 1294.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 1414.90 | 1429.47 | 1429.73 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 1435.50 | 1426.56 | 1426.49 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1408.30 | 1424.34 | 1425.69 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 1452.70 | 1427.71 | 1426.85 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1424.60 | 1439.58 | 1441.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 1419.90 | 1432.83 | 1437.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1446.60 | 1426.64 | 1432.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1457.00 | 1432.71 | 1434.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 1447.90 | 1432.71 | 1434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1445.60 | 1435.29 | 1435.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 1441.00 | 1435.29 | 1435.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 1446.30 | 1435.64 | 1435.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 1446.30 | 1435.64 | 1435.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1449.20 | 1438.35 | 1436.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1426.30 | 1437.16 | 1436.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 1423.90 | 1434.51 | 1435.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 1416.60 | 1427.24 | 1431.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1454.00 | 1424.63 | 1428.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1441.30 | 1427.97 | 1429.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1436.00 | 1427.97 | 1429.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 1435.40 | 1429.45 | 1429.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 1433.40 | 1430.24 | 1430.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1433.40 | 1430.24 | 1430.16 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 1415.60 | 1427.31 | 1428.84 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1447.60 | 1430.04 | 1429.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1478.80 | 1447.29 | 1439.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1467.50 | 1469.27 | 1456.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1467.50 | 1469.27 | 1456.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1462.90 | 1468.00 | 1456.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 1468.90 | 1464.59 | 1458.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 1472.00 | 1465.89 | 1460.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 1452.70 | 1461.96 | 1459.56 | SL hit (close<static) qty=1.00 sl=1454.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1539.30 | 1574.86 | 1576.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1536.10 | 1562.48 | 1570.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1541.50 | 1534.55 | 1551.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1541.50 | 1534.55 | 1551.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1555.70 | 1538.78 | 1551.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 1555.70 | 1538.78 | 1551.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1541.30 | 1539.28 | 1550.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 1531.30 | 1539.28 | 1550.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 1540.00 | 1539.05 | 1548.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1583.00 | 1556.28 | 1553.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1583.00 | 1556.28 | 1553.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1605.20 | 1581.80 | 1571.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 1579.20 | 1586.18 | 1578.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1579.20 | 1586.18 | 1578.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1580.90 | 1585.12 | 1578.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1585.10 | 1585.12 | 1578.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1594.00 | 1584.60 | 1578.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 1575.40 | 1582.76 | 1578.59 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1549.70 | 1571.19 | 1573.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1538.80 | 1564.71 | 1570.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1541.50 | 1537.78 | 1549.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1537.90 | 1519.49 | 1530.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 1547.10 | 1519.49 | 1530.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1568.10 | 1529.21 | 1534.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 1568.10 | 1529.21 | 1534.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 1572.00 | 1543.96 | 1540.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1574.20 | 1555.48 | 1547.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 1613.50 | 1617.12 | 1593.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 1613.50 | 1617.12 | 1593.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1603.40 | 1614.38 | 1594.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1604.80 | 1614.38 | 1594.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1634.80 | 1643.75 | 1632.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1632.00 | 1643.75 | 1632.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1637.10 | 1642.42 | 1632.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1639.30 | 1634.90 | 1632.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 1640.90 | 1634.26 | 1632.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 1647.80 | 1637.41 | 1633.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1642.30 | 1649.97 | 1650.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1642.30 | 1649.97 | 1650.09 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1663.70 | 1651.76 | 1650.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 1681.20 | 1664.69 | 1659.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1638.50 | 1656.53 | 1658.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1626.90 | 1645.11 | 1651.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1582.20 | 1578.15 | 1600.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 1585.20 | 1578.15 | 1600.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1574.20 | 1573.63 | 1589.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1587.70 | 1573.63 | 1589.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1558.50 | 1554.24 | 1567.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1564.90 | 1554.24 | 1567.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1555.20 | 1554.44 | 1566.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 1563.00 | 1554.44 | 1566.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1559.80 | 1555.64 | 1562.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1556.00 | 1555.64 | 1562.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1550.30 | 1554.57 | 1561.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1546.10 | 1552.90 | 1559.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1564.80 | 1550.10 | 1554.65 | SL hit (close>static) qty=1.00 sl=1563.80 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1561.90 | 1557.32 | 1556.80 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 1552.10 | 1556.28 | 1556.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1546.00 | 1554.22 | 1555.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 1553.00 | 1542.60 | 1547.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1547.70 | 1543.62 | 1547.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 1547.70 | 1543.62 | 1547.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1534.20 | 1541.74 | 1546.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 1545.00 | 1541.74 | 1546.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1536.90 | 1532.63 | 1539.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1536.90 | 1532.63 | 1539.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1532.30 | 1532.56 | 1538.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 1528.00 | 1532.56 | 1538.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1528.80 | 1533.76 | 1536.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 1528.60 | 1533.76 | 1536.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 1452.36 | 1476.47 | 1497.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 1452.17 | 1476.47 | 1497.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 1451.60 | 1469.57 | 1492.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1461.20 | 1453.23 | 1473.56 | SL hit (close>ema200) qty=0.50 sl=1453.23 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1432.80 | 1426.87 | 1426.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 11:15:00 | 1449.00 | 1431.30 | 1428.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 1510.70 | 1512.96 | 1489.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 1509.10 | 1512.96 | 1489.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1526.20 | 1533.23 | 1524.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1526.20 | 1533.23 | 1524.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1513.30 | 1529.24 | 1523.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1513.30 | 1529.24 | 1523.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1516.30 | 1526.65 | 1522.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1523.00 | 1526.65 | 1522.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 1527.90 | 1526.90 | 1523.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1513.10 | 1523.05 | 1523.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 1513.10 | 1523.05 | 1523.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1502.10 | 1518.86 | 1521.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1521.00 | 1518.31 | 1520.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1512.80 | 1517.21 | 1519.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 1518.40 | 1517.21 | 1519.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1524.40 | 1518.65 | 1520.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1524.40 | 1518.65 | 1520.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1521.00 | 1519.12 | 1520.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1498.40 | 1519.12 | 1520.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 1510.00 | 1508.47 | 1512.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 12:15:00 | 1434.50 | 1462.30 | 1482.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 14:15:00 | 1423.48 | 1447.60 | 1471.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1455.30 | 1435.44 | 1447.92 | SL hit (close>ema200) qty=0.50 sl=1435.44 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1209.90 | 1196.17 | 1196.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 1221.60 | 1209.99 | 1204.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1206.90 | 1215.07 | 1209.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1210.70 | 1214.20 | 1210.02 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1196.70 | 1206.09 | 1207.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1186.80 | 1200.78 | 1204.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1147.60 | 1145.68 | 1162.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 1153.50 | 1145.68 | 1162.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1157.30 | 1148.00 | 1161.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1157.10 | 1148.00 | 1161.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1157.80 | 1149.21 | 1158.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1157.80 | 1149.21 | 1158.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1175.60 | 1154.49 | 1160.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1175.60 | 1154.49 | 1160.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1176.70 | 1158.93 | 1161.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1206.40 | 1158.93 | 1161.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1193.00 | 1165.74 | 1164.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 1261.00 | 1211.67 | 1195.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 1235.20 | 1237.33 | 1219.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1235.20 | 1237.33 | 1219.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1217.80 | 1232.52 | 1224.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:15:00 | 1211.10 | 1232.52 | 1224.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1209.00 | 1227.82 | 1223.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1212.60 | 1227.82 | 1223.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1202.00 | 1219.45 | 1220.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1197.90 | 1210.06 | 1215.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 1213.60 | 1210.09 | 1214.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1209.20 | 1209.91 | 1214.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1198.50 | 1207.63 | 1212.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:45:00 | 1200.00 | 1205.90 | 1211.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1199.40 | 1207.17 | 1210.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 1198.00 | 1206.17 | 1209.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1204.20 | 1202.87 | 1206.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1262.10 | 1214.12 | 1209.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1262.10 | 1214.12 | 1209.57 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1220.70 | 1230.74 | 1231.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1211.70 | 1226.93 | 1229.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1220.00 | 1212.50 | 1218.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1207.90 | 1211.37 | 1216.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 1200.00 | 1208.62 | 1215.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1140.00 | 1162.41 | 1180.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1144.50 | 1142.70 | 1160.66 | SL hit (close>ema200) qty=0.50 sl=1142.70 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1055.30 | 1017.94 | 1013.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 1082.00 | 1030.75 | 1019.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1052.60 | 1055.84 | 1038.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:45:00 | 1051.20 | 1055.84 | 1038.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1083.00 | 1063.50 | 1049.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 1095.70 | 1077.80 | 1064.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 1092.60 | 1085.34 | 1075.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 12:15:00 | 1064.80 | 1071.24 | 1071.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 1064.80 | 1071.24 | 1071.63 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1079.30 | 1072.20 | 1071.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 1086.00 | 1074.96 | 1073.24 | Break + close above crossover candle high |

### Cycle 48 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 1060.50 | 1072.07 | 1072.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 1049.10 | 1058.69 | 1063.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1044.00 | 1030.78 | 1040.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1046.20 | 1033.87 | 1040.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 1046.50 | 1033.87 | 1040.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 1057.30 | 1045.91 | 1044.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1068.50 | 1051.88 | 1047.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 1061.00 | 1061.16 | 1055.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 1051.10 | 1061.16 | 1055.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1050.00 | 1058.93 | 1054.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1047.40 | 1058.93 | 1054.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1048.80 | 1056.90 | 1054.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1048.80 | 1056.90 | 1054.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1050.80 | 1053.12 | 1052.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 1049.70 | 1053.12 | 1052.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1051.70 | 1052.84 | 1052.84 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 1053.40 | 1052.95 | 1052.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1037.30 | 1049.82 | 1051.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1014.70 | 1032.51 | 1040.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1027.90 | 1026.39 | 1032.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 1027.90 | 1026.39 | 1032.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1040.00 | 1029.11 | 1033.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 1040.00 | 1029.11 | 1033.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1026.50 | 1028.59 | 1032.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:30:00 | 1025.50 | 1027.28 | 1031.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1024.40 | 1027.12 | 1030.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1025.00 | 1026.54 | 1028.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 1025.40 | 1026.54 | 1028.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1025.40 | 1026.31 | 1028.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1038.50 | 1026.31 | 1028.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1032.30 | 1027.51 | 1028.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1030.40 | 1029.39 | 1029.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1030.40 | 1029.39 | 1029.31 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1017.30 | 1026.98 | 1028.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1013.00 | 1024.18 | 1026.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 1019.00 | 1014.89 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1012.00 | 1014.32 | 1017.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1018.00 | 1014.32 | 1017.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 999.00 | 1011.25 | 1016.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 994.60 | 1008.14 | 1014.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1043.50 | 1011.72 | 1014.10 | SL hit (close>static) qty=1.00 sl=1020.60 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1041.00 | 1017.58 | 1016.54 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 1010.00 | 1018.15 | 1018.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 1008.70 | 1016.26 | 1017.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1010.80 | 1015.26 | 1016.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 1008.50 | 1014.00 | 1015.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 960.26 | 997.89 | 1005.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 958.07 | 997.89 | 1005.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 11:15:00 | 909.72 | 941.00 | 966.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 930.00 | 914.12 | 912.46 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 900.75 | 912.35 | 912.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 880.60 | 903.79 | 908.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 909.95 | 903.82 | 907.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 909.00 | 904.86 | 907.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 902.30 | 905.41 | 907.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 896.05 | 886.61 | 890.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:30:00 | 903.30 | 892.71 | 893.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 905.00 | 895.17 | 894.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 905.00 | 895.17 | 894.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 974.85 | 915.82 | 904.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 14:15:00 | 997.40 | 1003.84 | 976.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:45:00 | 996.50 | 1003.84 | 976.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 1032.10 | 1056.72 | 1033.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:45:00 | 1032.50 | 1056.72 | 1033.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 1023.85 | 1050.14 | 1032.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 1023.85 | 1050.14 | 1032.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 1030.50 | 1046.21 | 1032.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:30:00 | 1026.60 | 1046.21 | 1032.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1023.00 | 1038.65 | 1031.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1097.05 | 1038.65 | 1031.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 1028.35 | 1055.46 | 1057.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1028.35 | 1055.46 | 1057.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1019.45 | 1036.76 | 1046.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1044.65 | 996.14 | 1013.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1029.30 | 1002.77 | 1015.01 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1025.75 | 1021.90 | 1021.64 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 993.70 | 1017.19 | 1019.61 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1025.80 | 1017.96 | 1017.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1029.00 | 1020.16 | 1018.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 1026.75 | 1027.92 | 1024.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:15:00 | 1026.00 | 1027.92 | 1024.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1026.00 | 1027.54 | 1024.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1061.00 | 1027.54 | 1024.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 10:15:00 | 1167.10 | 1135.64 | 1102.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1201.20 | 1210.64 | 1211.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 1195.65 | 1207.64 | 1209.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 1216.25 | 1206.70 | 1208.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1230.00 | 1211.36 | 1210.62 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1202.50 | 1220.06 | 1221.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1200.00 | 1213.80 | 1218.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1222.95 | 1209.82 | 1214.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1223.40 | 1212.54 | 1215.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1225.80 | 1212.54 | 1215.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1227.00 | 1218.50 | 1217.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1245.10 | 1225.98 | 1221.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1264.40 | 1274.37 | 1260.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1256.20 | 1270.73 | 1260.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1256.20 | 1270.73 | 1260.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1264.00 | 1269.39 | 1260.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1259.05 | 1269.39 | 1260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1238.00 | 1263.11 | 1258.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 1236.05 | 1263.11 | 1258.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1232.15 | 1256.92 | 1256.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1229.25 | 1256.92 | 1256.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 1241.30 | 1253.79 | 1254.73 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1260.60 | 1250.03 | 1248.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1269.50 | 1257.97 | 1253.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 11:15:00 | 1314.40 | 2025-05-20 15:15:00 | 1263.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-06-11 13:15:00 | 1216.10 | 2025-06-19 11:15:00 | 1155.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1215.20 | 2025-06-19 11:15:00 | 1155.77 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1216.60 | 2025-06-19 12:15:00 | 1154.44 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-06-11 13:15:00 | 1216.10 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1215.20 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1216.60 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1209.70 | 2025-07-14 11:15:00 | 1217.10 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1266.50 | 2025-07-24 09:15:00 | 1393.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-18 14:30:00 | 1279.40 | 2025-07-24 09:15:00 | 1393.70 | TARGET_HIT | 1.00 | 8.93% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1268.00 | 2025-07-24 12:15:00 | 1407.34 | TARGET_HIT | 1.00 | 10.99% |
| BUY | retest2 | 2025-07-21 15:15:00 | 1267.00 | 2025-07-24 12:15:00 | 1394.80 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2025-07-22 09:15:00 | 1283.30 | 2025-07-24 12:15:00 | 1411.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 10:45:00 | 1272.40 | 2025-07-24 12:15:00 | 1399.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 12:00:00 | 1271.50 | 2025-07-24 12:15:00 | 1398.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 12:45:00 | 1273.00 | 2025-07-24 12:15:00 | 1400.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-07 12:15:00 | 1441.00 | 2025-08-07 13:15:00 | 1446.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1436.00 | 2025-08-11 13:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-11 13:00:00 | 1435.40 | 2025-08-11 13:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-08-14 14:45:00 | 1468.90 | 2025-08-18 12:15:00 | 1452.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-18 10:30:00 | 1472.00 | 2025-08-18 12:15:00 | 1452.70 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-19 09:30:00 | 1476.80 | 2025-08-25 11:15:00 | 1624.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 13:15:00 | 1531.30 | 2025-09-01 14:15:00 | 1583.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-08-29 15:15:00 | 1540.00 | 2025-09-01 14:15:00 | 1583.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1585.10 | 2025-09-04 10:15:00 | 1575.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1594.00 | 2025-09-04 10:15:00 | 1575.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1639.30 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-09-17 10:15:00 | 1640.90 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-17 10:45:00 | 1647.80 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1546.10 | 2025-10-06 09:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-09 13:15:00 | 1528.00 | 2025-10-14 10:15:00 | 1452.36 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1528.80 | 2025-10-14 10:15:00 | 1452.17 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1528.60 | 2025-10-14 11:15:00 | 1451.60 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-10-09 13:15:00 | 1528.00 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1528.80 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1528.60 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.41% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1523.00 | 2025-11-04 15:15:00 | 1513.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-11-04 10:00:00 | 1527.90 | 2025-11-04 15:15:00 | 1513.10 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1498.40 | 2025-11-11 12:15:00 | 1434.50 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1510.00 | 2025-11-11 14:15:00 | 1423.48 | PARTIAL | 0.50 | 5.73% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1498.40 | 2025-11-13 09:15:00 | 1455.30 | STOP_HIT | 0.50 | 2.88% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1510.00 | 2025-11-13 09:15:00 | 1455.30 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1198.50 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-12-30 12:45:00 | 1200.00 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2025-12-31 09:45:00 | 1199.40 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2025-12-31 12:15:00 | 1198.00 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-01-08 10:45:00 | 1200.00 | 2026-01-12 09:15:00 | 1140.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 1200.00 | 2026-01-12 15:15:00 | 1144.50 | STOP_HIT | 0.50 | 4.63% |
| BUY | retest2 | 2026-02-01 09:15:00 | 1095.70 | 2026-02-02 12:15:00 | 1064.80 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-02-01 14:30:00 | 1092.60 | 2026-02-02 12:15:00 | 1064.80 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-02-16 13:30:00 | 1025.50 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-02-17 11:00:00 | 1024.40 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1025.00 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1025.40 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-02-23 10:30:00 | 994.60 | 2026-02-23 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1010.80 | 2026-03-02 09:15:00 | 960.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 1008.50 | 2026-03-02 09:15:00 | 958.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1010.80 | 2026-03-04 11:15:00 | 909.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 1008.50 | 2026-03-04 13:15:00 | 907.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 902.30 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-03-16 15:00:00 | 896.05 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-17 09:30:00 | 903.30 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1097.05 | 2026-03-27 10:15:00 | 1028.35 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1061.00 | 2026-04-10 10:15:00 | 1167.10 | TARGET_HIT | 1.00 | 10.00% |
