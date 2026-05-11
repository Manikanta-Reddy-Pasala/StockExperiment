# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 722.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 48 |
| ALERT1 | 32 |
| ALERT2 | 33 |
| ALERT2_SKIP | 17 |
| ALERT3 | 73 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 12 |
| TARGET_HIT | 1 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 17
- **Target hits / Stop hits / Partials:** 1 / 40 / 12
- **Avg / median % per leg:** 2.02% / 2.53%
- **Sum % (uncompounded):** 107.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 0 | 17 | 0 | -0.83% | -14.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 0 | 17 | 0 | -0.83% | -14.2% |
| SELL (all) | 36 | 30 | 83.3% | 1 | 23 | 12 | 3.37% | 121.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 30 | 83.3% | 1 | 23 | 12 | 3.37% | 121.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 36 | 67.9% | 1 | 40 | 12 | 2.02% | 107.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1211.80 | 1205.69 | 1204.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1214.20 | 1207.39 | 1205.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1209.90 | 1212.00 | 1208.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 1211.80 | 1212.00 | 1208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1208.00 | 1211.20 | 1208.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 1206.10 | 1211.20 | 1208.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1205.00 | 1209.96 | 1208.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 1205.00 | 1209.96 | 1208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1202.50 | 1208.47 | 1207.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 1204.80 | 1208.47 | 1207.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 15:15:00 | 1203.10 | 1207.39 | 1207.47 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1209.60 | 1207.83 | 1207.67 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 1205.20 | 1207.56 | 1207.59 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 1211.60 | 1208.37 | 1207.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 1220.00 | 1214.02 | 1211.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 1248.00 | 1248.02 | 1238.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 1248.00 | 1248.02 | 1238.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1242.10 | 1246.55 | 1240.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1238.60 | 1246.55 | 1240.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1242.00 | 1245.64 | 1240.50 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1224.00 | 1235.81 | 1237.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1216.00 | 1231.85 | 1235.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1241.90 | 1231.43 | 1234.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1241.90 | 1231.43 | 1234.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1242.70 | 1233.69 | 1234.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:15:00 | 1247.50 | 1233.69 | 1234.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 1253.80 | 1237.71 | 1236.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 1261.20 | 1255.31 | 1249.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1270.20 | 1278.66 | 1271.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 1268.30 | 1278.66 | 1271.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1280.30 | 1278.99 | 1272.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 1285.00 | 1279.59 | 1273.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1263.50 | 1276.37 | 1274.30 | SL hit (close<static) qty=1.00 sl=1268.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1265.00 | 1273.50 | 1274.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 1261.80 | 1271.16 | 1273.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 1264.50 | 1262.66 | 1267.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:30:00 | 1263.90 | 1262.66 | 1267.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1265.10 | 1264.28 | 1266.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:45:00 | 1268.20 | 1264.28 | 1266.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1266.00 | 1264.63 | 1266.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 1266.00 | 1264.63 | 1266.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1260.00 | 1263.70 | 1265.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:30:00 | 1254.80 | 1261.66 | 1264.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1255.90 | 1261.02 | 1263.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1223.90 | 1220.46 | 1220.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 10:15:00 | 1223.90 | 1220.46 | 1220.25 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1213.00 | 1219.46 | 1219.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1210.00 | 1214.94 | 1217.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1212.90 | 1211.69 | 1213.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 1212.90 | 1211.69 | 1213.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1217.30 | 1212.81 | 1214.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1217.30 | 1212.81 | 1214.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1216.00 | 1213.45 | 1214.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1215.60 | 1213.45 | 1214.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1222.00 | 1216.37 | 1215.63 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1209.10 | 1215.20 | 1215.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1207.50 | 1210.58 | 1212.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1206.30 | 1206.18 | 1208.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1206.30 | 1206.18 | 1208.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1200.20 | 1204.99 | 1207.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:00:00 | 1197.80 | 1201.25 | 1204.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:00:00 | 1196.30 | 1199.58 | 1203.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 1211.20 | 1203.59 | 1203.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1211.20 | 1203.59 | 1203.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 15:15:00 | 1215.00 | 1210.33 | 1207.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1210.40 | 1210.55 | 1207.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:00:00 | 1210.40 | 1210.55 | 1207.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1205.80 | 1209.60 | 1207.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 1205.90 | 1209.60 | 1207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1207.10 | 1209.10 | 1207.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 1223.60 | 1208.88 | 1207.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 1225.00 | 1234.26 | 1234.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 1225.00 | 1234.26 | 1234.33 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1259.90 | 1239.39 | 1236.65 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1245.00 | 1255.17 | 1255.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1237.20 | 1251.58 | 1253.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1246.90 | 1240.71 | 1244.02 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1258.00 | 1248.13 | 1247.07 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 1244.50 | 1248.87 | 1249.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1235.00 | 1245.16 | 1247.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 15:15:00 | 1210.80 | 1210.72 | 1215.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:15:00 | 1207.10 | 1210.72 | 1215.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1210.50 | 1207.10 | 1210.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 1212.00 | 1207.10 | 1210.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1211.60 | 1208.00 | 1210.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 1211.00 | 1208.00 | 1210.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1208.60 | 1208.12 | 1210.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 10:00:00 | 1205.60 | 1209.06 | 1210.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 15:15:00 | 1203.90 | 1207.64 | 1208.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 1186.50 | 1183.44 | 1183.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 1186.50 | 1183.44 | 1183.30 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 1163.50 | 1180.68 | 1182.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1149.00 | 1166.55 | 1174.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1115.70 | 1077.91 | 1100.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1115.70 | 1077.91 | 1100.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1120.90 | 1086.51 | 1102.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 1122.00 | 1086.51 | 1102.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 1113.30 | 1109.07 | 1108.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 1131.70 | 1116.38 | 1112.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 10:15:00 | 1130.00 | 1131.14 | 1122.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:00:00 | 1130.00 | 1131.14 | 1122.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1122.00 | 1128.32 | 1124.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1123.60 | 1128.32 | 1124.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1123.10 | 1127.28 | 1124.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:15:00 | 1120.90 | 1127.28 | 1124.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1122.60 | 1124.57 | 1124.02 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1115.50 | 1122.10 | 1122.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1108.30 | 1119.34 | 1121.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 11:15:00 | 1075.90 | 1075.24 | 1085.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:45:00 | 1076.00 | 1075.24 | 1085.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1086.40 | 1078.58 | 1083.30 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 1106.10 | 1086.66 | 1086.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1136.50 | 1115.92 | 1105.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 10:15:00 | 1238.00 | 1238.71 | 1219.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:30:00 | 1234.50 | 1238.71 | 1219.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1222.00 | 1235.37 | 1220.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 1222.00 | 1235.37 | 1220.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1221.40 | 1232.57 | 1220.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:30:00 | 1226.10 | 1231.54 | 1220.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1206.70 | 1225.24 | 1220.61 | SL hit (close<static) qty=1.00 sl=1220.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 1257.60 | 1260.88 | 1261.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 1252.00 | 1259.11 | 1260.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1225.60 | 1224.90 | 1232.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1225.60 | 1224.90 | 1232.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1174.70 | 1165.48 | 1173.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 1174.70 | 1165.48 | 1173.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1167.10 | 1165.81 | 1172.93 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1189.00 | 1175.61 | 1174.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1189.20 | 1178.32 | 1176.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1216.90 | 1219.39 | 1212.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 1216.90 | 1219.39 | 1212.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1207.30 | 1216.28 | 1212.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1207.30 | 1216.28 | 1212.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1207.90 | 1214.61 | 1211.87 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1201.80 | 1209.05 | 1209.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 1200.20 | 1207.28 | 1208.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1128.90 | 1126.22 | 1136.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 1143.00 | 1126.22 | 1136.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1142.40 | 1129.45 | 1137.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 1143.40 | 1129.45 | 1137.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1143.20 | 1132.20 | 1137.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:15:00 | 1144.30 | 1132.20 | 1137.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1148.00 | 1139.44 | 1139.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1148.00 | 1139.44 | 1139.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 1151.00 | 1141.75 | 1140.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1158.40 | 1145.81 | 1142.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1145.20 | 1147.32 | 1144.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 1145.20 | 1147.32 | 1144.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1146.30 | 1147.11 | 1144.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1145.10 | 1147.11 | 1144.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1154.00 | 1149.61 | 1146.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 1165.00 | 1152.67 | 1147.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 1162.00 | 1166.30 | 1164.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1167.00 | 1166.68 | 1165.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 1162.00 | 1165.64 | 1165.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1168.90 | 1166.29 | 1165.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 1164.90 | 1166.29 | 1165.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1165.30 | 1166.09 | 1165.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 1165.30 | 1166.09 | 1165.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1165.20 | 1165.91 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:15:00 | 1166.00 | 1165.91 | 1165.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1166.90 | 1166.11 | 1165.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1167.20 | 1166.11 | 1165.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1168.20 | 1166.53 | 1165.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1102.00 | 1166.53 | 1165.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1112.30 | 1155.68 | 1160.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1112.30 | 1155.68 | 1160.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 1090.00 | 1121.79 | 1139.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 1072.10 | 1070.82 | 1078.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 1072.10 | 1070.82 | 1078.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1070.40 | 1070.45 | 1075.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 1067.80 | 1069.47 | 1074.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1058.40 | 1050.74 | 1050.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1058.40 | 1050.74 | 1050.52 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 1049.80 | 1050.37 | 1050.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1044.90 | 1049.15 | 1049.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1015.00 | 1012.88 | 1019.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 1009.50 | 1012.88 | 1019.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1010.00 | 1012.35 | 1018.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:45:00 | 1010.10 | 1011.86 | 1018.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 1008.90 | 1011.57 | 1017.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1014.10 | 1012.18 | 1015.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 1009.90 | 1012.16 | 1014.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 959.50 | 967.15 | 976.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 959.60 | 967.15 | 976.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 11:15:00 | 967.25 | 966.64 | 974.59 | SL hit (close>ema200) qty=0.50 sl=966.64 alert=retest2 |

### Cycle 31 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 982.50 | 962.66 | 961.63 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 11:15:00 | 962.40 | 969.96 | 969.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 09:15:00 | 958.00 | 960.45 | 963.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 960.35 | 960.13 | 962.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:45:00 | 960.00 | 960.13 | 962.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 941.40 | 939.60 | 943.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 943.00 | 939.60 | 943.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 943.75 | 940.43 | 943.25 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 952.15 | 944.29 | 944.18 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 937.25 | 947.10 | 947.93 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 945.00 | 942.73 | 942.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 947.35 | 943.66 | 943.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 944.60 | 946.10 | 944.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 944.60 | 946.10 | 944.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 946.05 | 946.09 | 945.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:00:00 | 948.70 | 946.61 | 945.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:45:00 | 948.35 | 946.96 | 945.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 947.95 | 947.72 | 946.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:00:00 | 948.50 | 951.30 | 950.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 946.75 | 949.69 | 949.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 946.75 | 949.69 | 949.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 938.60 | 947.15 | 948.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 928.50 | 927.16 | 932.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 929.70 | 927.16 | 932.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 894.70 | 893.86 | 898.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 898.20 | 893.86 | 898.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 891.15 | 893.32 | 898.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:45:00 | 889.40 | 892.50 | 897.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 887.95 | 891.01 | 896.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 878.70 | 889.88 | 894.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 844.93 | 861.77 | 871.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 843.55 | 861.77 | 871.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 859.60 | 851.69 | 859.88 | SL hit (close>ema200) qty=0.50 sl=851.69 alert=retest2 |

### Cycle 37 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 870.60 | 857.68 | 856.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 873.30 | 864.08 | 860.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 866.10 | 866.70 | 863.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:45:00 | 865.20 | 866.70 | 863.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 865.00 | 866.36 | 863.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 856.65 | 866.36 | 863.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 856.65 | 864.42 | 862.79 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 850.50 | 859.88 | 860.90 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 872.40 | 862.31 | 861.39 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 860.90 | 863.46 | 863.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 855.10 | 861.23 | 862.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 862.35 | 851.86 | 853.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 863.75 | 851.86 | 853.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 867.45 | 854.98 | 855.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 867.45 | 854.98 | 855.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 869.75 | 857.93 | 856.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 874.30 | 861.21 | 857.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 906.15 | 909.01 | 892.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 888.65 | 902.55 | 897.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 888.65 | 902.55 | 897.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 888.65 | 902.55 | 897.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 885.85 | 899.21 | 896.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 885.85 | 899.21 | 896.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 885.85 | 893.95 | 894.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 880.05 | 888.83 | 891.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 839.90 | 839.24 | 854.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 839.90 | 839.24 | 854.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 837.35 | 838.63 | 846.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 833.70 | 838.63 | 846.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:45:00 | 834.70 | 836.24 | 842.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:15:00 | 835.40 | 836.24 | 842.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 835.00 | 836.69 | 841.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 806.80 | 799.86 | 804.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 806.40 | 799.86 | 804.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 803.05 | 800.50 | 804.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 796.65 | 800.50 | 804.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 11:15:00 | 793.63 | 799.70 | 803.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 792.97 | 797.13 | 800.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 15:15:00 | 793.25 | 797.13 | 800.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 797.60 | 797.22 | 800.52 | SL hit (close>ema200) qty=0.50 sl=797.22 alert=retest2 |

### Cycle 43 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 659.75 | 646.91 | 645.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 664.30 | 654.41 | 649.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 640.15 | 651.56 | 648.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 640.15 | 651.56 | 648.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 640.70 | 649.39 | 648.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 640.70 | 649.39 | 648.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 644.00 | 647.13 | 647.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 636.00 | 643.87 | 645.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 636.20 | 621.04 | 629.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 636.20 | 621.04 | 629.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 629.70 | 622.77 | 629.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 628.35 | 628.78 | 630.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:00:00 | 628.30 | 628.78 | 630.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 640.95 | 630.83 | 630.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 640.95 | 630.83 | 630.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 15:15:00 | 659.00 | 642.85 | 636.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 642.05 | 642.69 | 637.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 723.85 | 714.19 | 703.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 723.85 | 714.19 | 703.87 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 756.00 | 761.64 | 761.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 747.85 | 758.88 | 760.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 728.70 | 728.60 | 734.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 723.50 | 721.48 | 725.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 723.50 | 721.48 | 725.31 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 724.50 | 722.71 | 722.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 731.95 | 724.56 | 723.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 724.55 | 726.09 | 724.63 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 721.95 | 724.24 | 724.43 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:45:00 | 1209.10 | 2025-05-12 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-12 12:45:00 | 1208.40 | 2025-05-12 13:15:00 | 1211.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-27 11:30:00 | 1285.00 | 2025-05-28 09:15:00 | 1263.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1284.10 | 2025-05-29 10:15:00 | 1265.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-06-02 12:30:00 | 1254.80 | 2025-06-12 10:15:00 | 1223.90 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2025-06-03 09:15:00 | 1255.90 | 2025-06-12 10:15:00 | 1223.90 | STOP_HIT | 1.00 | 2.55% |
| SELL | retest2 | 2025-06-23 13:00:00 | 1197.80 | 2025-06-25 09:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-06-23 15:00:00 | 1196.30 | 2025-06-25 09:15:00 | 1211.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-26 14:15:00 | 1223.60 | 2025-07-07 15:15:00 | 1225.00 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-25 10:00:00 | 1205.60 | 2025-08-11 11:15:00 | 1186.50 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2025-07-25 15:15:00 | 1203.90 | 2025-08-11 11:15:00 | 1186.50 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-09-08 13:30:00 | 1226.10 | 2025-09-09 09:15:00 | 1206.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1233.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-09-09 11:15:00 | 1226.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-09-09 12:30:00 | 1225.60 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1245.50 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-09-12 09:45:00 | 1247.00 | 2025-09-18 10:15:00 | 1257.60 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2025-10-20 10:45:00 | 1165.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2025-10-24 14:15:00 | 1162.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-10-27 09:30:00 | 1167.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.69% |
| BUY | retest2 | 2025-10-27 10:45:00 | 1162.00 | 2025-10-28 09:15:00 | 1112.30 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-11-04 11:30:00 | 1067.80 | 2025-11-12 15:15:00 | 1058.40 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1009.50 | 2025-12-03 09:15:00 | 959.50 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1010.00 | 2025-12-03 09:15:00 | 959.60 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-11-19 09:15:00 | 1009.50 | 2025-12-03 11:15:00 | 967.25 | STOP_HIT | 0.50 | 4.19% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1010.00 | 2025-12-03 11:15:00 | 967.25 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2025-11-19 10:45:00 | 1010.10 | 2025-12-04 14:15:00 | 959.02 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-19 11:45:00 | 1008.90 | 2025-12-04 14:15:00 | 958.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:45:00 | 1009.90 | 2025-12-04 14:15:00 | 959.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 10:45:00 | 1010.10 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-11-19 11:45:00 | 1008.90 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2025-11-20 11:45:00 | 1009.90 | 2025-12-05 11:15:00 | 965.00 | STOP_HIT | 0.50 | 4.45% |
| BUY | retest2 | 2026-01-02 12:00:00 | 948.70 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-02 12:45:00 | 948.35 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-02 14:45:00 | 947.95 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-01-06 13:00:00 | 948.50 | 2026-01-06 14:15:00 | 946.75 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-01-20 12:45:00 | 889.40 | 2026-01-27 09:15:00 | 844.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 13:30:00 | 887.95 | 2026-01-27 09:15:00 | 843.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:45:00 | 889.40 | 2026-01-28 09:15:00 | 859.60 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-01-20 13:30:00 | 887.95 | 2026-01-28 09:15:00 | 859.60 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-01-21 09:15:00 | 878.70 | 2026-01-30 10:15:00 | 870.60 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2026-02-18 10:15:00 | 833.70 | 2026-02-26 11:15:00 | 793.63 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2026-02-18 13:45:00 | 834.70 | 2026-02-26 15:15:00 | 792.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:15:00 | 835.40 | 2026-02-26 15:15:00 | 793.25 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-02-18 10:15:00 | 833.70 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2026-02-18 13:45:00 | 834.70 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-18 14:15:00 | 835.40 | 2026-02-27 09:15:00 | 797.60 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2026-02-19 09:15:00 | 835.00 | 2026-02-27 10:15:00 | 792.01 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2026-02-26 11:15:00 | 796.65 | 2026-03-04 09:15:00 | 756.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 835.00 | 2026-03-04 10:15:00 | 750.33 | TARGET_HIT | 0.50 | 10.14% |
| SELL | retest2 | 2026-02-26 11:15:00 | 796.65 | 2026-03-05 11:15:00 | 743.15 | STOP_HIT | 0.50 | 6.72% |
| SELL | retest2 | 2026-04-01 14:30:00 | 628.35 | 2026-04-02 12:15:00 | 640.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-04-01 15:00:00 | 628.30 | 2026-04-02 12:15:00 | 640.95 | STOP_HIT | 1.00 | -2.01% |
