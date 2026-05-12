# Signatureglobal (India) Ltd. (SIGNATURE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 885.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 28 |
| ALERT3 | 145 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 77 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 36 / 50
- **Target hits / Stop hits / Partials:** 3 / 72 / 11
- **Avg / median % per leg:** 1.20% / -0.28%
- **Sum % (uncompounded):** 103.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 7 | 20.6% | 0 | 34 | 0 | -0.68% | -23.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 7 | 20.6% | 0 | 34 | 0 | -0.68% | -23.1% |
| SELL (all) | 52 | 29 | 55.8% | 3 | 38 | 11 | 2.43% | 126.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 29 | 55.8% | 3 | 38 | 11 | 2.43% | 126.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 86 | 36 | 41.9% | 3 | 72 | 11 | 1.20% | 103.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 1203.00 | 1228.35 | 1228.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 13:15:00 | 1192.00 | 1205.64 | 1214.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1205.00 | 1203.89 | 1210.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 12:00:00 | 1205.00 | 1203.89 | 1210.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1192.00 | 1198.70 | 1205.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:30:00 | 1190.20 | 1197.40 | 1202.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1226.70 | 1202.46 | 1201.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1226.70 | 1202.46 | 1201.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 12:15:00 | 1227.90 | 1221.93 | 1217.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 1245.40 | 1246.98 | 1237.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 1245.40 | 1246.98 | 1237.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1239.30 | 1245.45 | 1237.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 1239.70 | 1245.45 | 1237.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1239.10 | 1244.18 | 1237.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 1238.60 | 1244.18 | 1237.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1235.40 | 1242.42 | 1237.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:15:00 | 1233.70 | 1242.42 | 1237.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1233.10 | 1240.56 | 1237.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1233.10 | 1240.56 | 1237.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1235.30 | 1237.90 | 1236.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:15:00 | 1239.40 | 1237.90 | 1236.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 11:45:00 | 1239.20 | 1239.66 | 1237.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 15:15:00 | 1275.00 | 1289.76 | 1290.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1275.00 | 1289.76 | 1290.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1261.70 | 1284.15 | 1288.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1289.90 | 1277.76 | 1283.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1289.90 | 1277.76 | 1283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1289.00 | 1280.00 | 1283.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 1282.00 | 1280.00 | 1283.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 1291.20 | 1285.43 | 1285.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 1291.20 | 1285.43 | 1285.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1294.00 | 1287.23 | 1286.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1285.30 | 1287.68 | 1286.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 1285.30 | 1287.68 | 1286.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1284.20 | 1286.98 | 1286.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1281.30 | 1286.98 | 1286.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 1276.10 | 1284.80 | 1285.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1273.80 | 1282.60 | 1284.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1282.30 | 1281.49 | 1283.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 1272.80 | 1280.35 | 1282.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1272.30 | 1277.96 | 1281.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 1272.80 | 1274.08 | 1278.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 1266.70 | 1268.90 | 1275.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1277.20 | 1266.22 | 1272.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:30:00 | 1271.40 | 1266.22 | 1272.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1271.00 | 1267.18 | 1271.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1270.00 | 1267.18 | 1271.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:45:00 | 1268.40 | 1267.54 | 1271.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 1269.60 | 1268.31 | 1271.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 1252.00 | 1246.85 | 1246.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 1252.00 | 1246.85 | 1246.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 1254.30 | 1250.65 | 1248.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1249.20 | 1250.83 | 1249.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1244.10 | 1250.83 | 1249.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1244.20 | 1249.51 | 1248.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 1247.20 | 1249.51 | 1248.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1250.50 | 1249.70 | 1249.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 12:30:00 | 1260.00 | 1250.44 | 1249.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 1240.20 | 1247.39 | 1248.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1240.20 | 1247.39 | 1248.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1237.80 | 1245.47 | 1247.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 1236.80 | 1236.50 | 1240.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 11:00:00 | 1236.80 | 1236.50 | 1240.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1243.90 | 1237.93 | 1240.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 1243.90 | 1237.93 | 1240.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1252.20 | 1240.79 | 1241.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:30:00 | 1253.50 | 1240.79 | 1241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1248.30 | 1242.29 | 1242.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 1255.30 | 1248.21 | 1245.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1243.00 | 1247.16 | 1245.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 1242.00 | 1247.16 | 1245.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1239.70 | 1245.67 | 1244.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:45:00 | 1245.00 | 1246.12 | 1244.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 1246.50 | 1244.98 | 1244.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 10:15:00 | 1246.50 | 1244.98 | 1244.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 11:15:00 | 1237.20 | 1243.49 | 1243.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 1237.20 | 1243.49 | 1243.98 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1250.50 | 1244.13 | 1243.74 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1241.40 | 1244.53 | 1244.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 1233.50 | 1241.79 | 1243.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1244.00 | 1240.05 | 1241.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1244.00 | 1240.05 | 1241.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1243.00 | 1240.64 | 1241.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1241.40 | 1240.64 | 1241.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 1242.40 | 1240.99 | 1241.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1255.90 | 1243.97 | 1243.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1255.90 | 1243.97 | 1243.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 1267.00 | 1253.65 | 1248.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1257.70 | 1260.95 | 1255.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1257.70 | 1260.95 | 1255.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1253.90 | 1259.54 | 1255.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1253.90 | 1259.54 | 1255.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1255.50 | 1258.73 | 1255.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 12:30:00 | 1256.10 | 1259.58 | 1255.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1256.50 | 1258.45 | 1255.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 1260.00 | 1258.45 | 1255.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1251.80 | 1256.56 | 1255.70 | SL hit (close<static) qty=1.00 sl=1252.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1247.70 | 1254.79 | 1254.97 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 1258.00 | 1254.68 | 1254.37 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 1250.90 | 1254.26 | 1254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 11:15:00 | 1248.00 | 1253.01 | 1254.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1254.40 | 1253.29 | 1254.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1254.40 | 1253.29 | 1254.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1245.00 | 1251.63 | 1253.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 1242.50 | 1250.60 | 1252.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1242.10 | 1249.88 | 1252.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1241.30 | 1248.17 | 1251.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1239.80 | 1244.99 | 1249.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1243.90 | 1244.26 | 1248.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 1245.60 | 1244.26 | 1248.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1246.20 | 1244.65 | 1247.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 1248.00 | 1244.65 | 1247.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1245.00 | 1244.72 | 1247.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1248.50 | 1244.72 | 1247.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1246.10 | 1245.00 | 1247.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1258.40 | 1245.00 | 1247.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1248.30 | 1245.66 | 1247.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1244.90 | 1245.66 | 1247.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1241.10 | 1244.75 | 1246.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1245.80 | 1244.75 | 1246.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1246.60 | 1245.12 | 1246.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1248.20 | 1245.12 | 1246.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1241.50 | 1244.39 | 1246.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 1239.30 | 1244.39 | 1246.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1234.90 | 1244.15 | 1246.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1180.38 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1179.99 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1179.23 | 1196.14 | 1207.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1177.81 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1177.33 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1173.15 | 1190.37 | 1201.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 1135.30 | 1135.10 | 1152.63 | SL hit (close>ema200) qty=0.50 sl=1135.10 alert=retest2 |

### Cycle 16 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1158.60 | 1140.01 | 1137.61 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1128.90 | 1139.33 | 1139.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 1124.70 | 1133.60 | 1136.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1127.90 | 1127.75 | 1132.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 1129.60 | 1127.75 | 1132.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1117.40 | 1115.04 | 1122.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 1106.20 | 1111.58 | 1118.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:15:00 | 1106.00 | 1112.04 | 1113.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1107.90 | 1109.67 | 1112.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1107.10 | 1108.32 | 1109.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1114.30 | 1104.86 | 1106.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1114.30 | 1104.86 | 1106.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1108.60 | 1105.61 | 1106.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:15:00 | 1104.20 | 1105.61 | 1106.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 1102.80 | 1106.01 | 1106.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 1101.90 | 1105.93 | 1106.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:00:00 | 1103.00 | 1104.23 | 1105.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1104.00 | 1104.18 | 1105.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 1102.80 | 1104.18 | 1105.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1101.80 | 1103.70 | 1105.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 1108.90 | 1106.15 | 1105.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 1108.90 | 1106.15 | 1105.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 1113.10 | 1107.54 | 1106.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1121.50 | 1125.58 | 1119.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1121.50 | 1125.58 | 1119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1116.00 | 1123.66 | 1119.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1113.50 | 1123.66 | 1119.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1118.50 | 1122.63 | 1118.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:00:00 | 1126.40 | 1122.42 | 1119.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1132.40 | 1124.23 | 1120.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1119.10 | 1122.76 | 1123.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1119.10 | 1122.76 | 1123.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1109.50 | 1116.86 | 1119.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 1088.00 | 1081.33 | 1089.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 1066.80 | 1081.33 | 1089.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1073.50 | 1079.77 | 1088.36 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1101.00 | 1087.49 | 1086.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 11:15:00 | 1115.30 | 1101.20 | 1095.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1102.00 | 1104.24 | 1098.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1102.00 | 1104.24 | 1098.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1097.30 | 1102.85 | 1098.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1095.00 | 1101.28 | 1098.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1094.00 | 1099.82 | 1097.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1094.00 | 1099.82 | 1097.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1091.20 | 1098.10 | 1097.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 1090.60 | 1098.10 | 1097.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1104.70 | 1107.85 | 1104.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1104.70 | 1107.85 | 1104.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1105.00 | 1107.28 | 1104.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1100.90 | 1107.28 | 1104.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1105.70 | 1106.96 | 1104.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 1115.10 | 1108.29 | 1105.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 1121.80 | 1111.51 | 1108.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 1117.40 | 1123.61 | 1122.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 1112.00 | 1120.30 | 1120.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 1112.00 | 1120.30 | 1120.71 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 10:15:00 | 1129.30 | 1122.40 | 1121.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1133.80 | 1125.46 | 1123.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 1134.00 | 1136.23 | 1130.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 1134.00 | 1136.23 | 1130.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1143.00 | 1136.89 | 1131.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1149.00 | 1138.09 | 1132.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 1144.20 | 1138.09 | 1133.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 1143.30 | 1139.13 | 1134.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:00:00 | 1143.20 | 1139.95 | 1135.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1134.50 | 1139.17 | 1135.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1134.50 | 1139.17 | 1135.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1131.00 | 1137.53 | 1135.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-17 15:15:00 | 1131.00 | 1137.53 | 1135.53 | SL hit (close<static) qty=1.00 sl=1131.20 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 1112.10 | 1133.47 | 1136.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 1106.70 | 1128.12 | 1133.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 1125.10 | 1124.61 | 1130.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:30:00 | 1125.30 | 1124.61 | 1130.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1125.90 | 1124.87 | 1130.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1122.10 | 1124.87 | 1130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1109.50 | 1110.50 | 1116.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:30:00 | 1101.40 | 1108.50 | 1115.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:15:00 | 1046.33 | 1057.30 | 1065.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1052.60 | 1051.06 | 1057.95 | SL hit (close>ema200) qty=0.50 sl=1051.06 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1025.20 | 1017.35 | 1016.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1032.00 | 1020.28 | 1018.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1020.90 | 1022.37 | 1019.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 1020.90 | 1022.37 | 1019.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1014.10 | 1026.30 | 1024.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1014.10 | 1026.30 | 1024.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1006.80 | 1022.40 | 1022.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1000.00 | 1017.92 | 1020.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1019.10 | 1012.18 | 1016.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1019.10 | 1012.18 | 1016.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1016.90 | 1013.12 | 1016.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1016.90 | 1013.12 | 1016.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1018.70 | 1014.24 | 1016.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 1018.70 | 1014.24 | 1016.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1030.00 | 1017.39 | 1017.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 1030.00 | 1017.39 | 1017.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1025.20 | 1018.95 | 1018.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1041.00 | 1023.36 | 1020.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1058.30 | 1059.10 | 1049.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 1058.30 | 1059.10 | 1049.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1060.10 | 1067.02 | 1064.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1060.10 | 1067.02 | 1064.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1061.00 | 1065.81 | 1064.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:45:00 | 1078.00 | 1068.67 | 1065.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1082.70 | 1092.66 | 1092.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1082.70 | 1092.66 | 1092.76 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1100.80 | 1093.67 | 1092.97 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 1086.50 | 1093.03 | 1093.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 1075.10 | 1087.60 | 1091.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1083.00 | 1081.23 | 1086.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 1083.00 | 1081.23 | 1086.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1040.70 | 1030.32 | 1041.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 1040.70 | 1030.32 | 1041.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1041.30 | 1032.52 | 1041.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:15:00 | 1040.10 | 1032.52 | 1041.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1040.10 | 1034.03 | 1041.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 1047.30 | 1034.03 | 1041.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1035.80 | 1034.39 | 1040.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 1033.70 | 1034.65 | 1040.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 1060.00 | 1043.47 | 1043.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 1060.00 | 1043.47 | 1043.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 1065.70 | 1047.91 | 1045.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 1102.00 | 1106.41 | 1094.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 11:45:00 | 1101.00 | 1106.41 | 1094.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1113.20 | 1110.79 | 1103.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 1106.30 | 1110.79 | 1103.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1102.70 | 1109.17 | 1105.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1102.70 | 1109.17 | 1105.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1102.40 | 1107.82 | 1104.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:45:00 | 1111.10 | 1108.97 | 1105.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1110.30 | 1114.76 | 1114.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1110.30 | 1114.76 | 1114.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 1100.60 | 1111.93 | 1113.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1111.80 | 1107.14 | 1110.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1111.80 | 1107.14 | 1110.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1112.50 | 1108.21 | 1110.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:15:00 | 1110.70 | 1108.21 | 1110.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1126.00 | 1112.17 | 1111.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 1131.00 | 1115.93 | 1113.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1125.80 | 1127.48 | 1122.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 1125.80 | 1127.48 | 1122.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1123.90 | 1126.34 | 1122.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1122.10 | 1126.34 | 1122.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1121.10 | 1125.29 | 1122.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1121.10 | 1125.29 | 1122.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1122.90 | 1124.81 | 1122.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:30:00 | 1121.10 | 1124.81 | 1122.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1121.10 | 1124.07 | 1122.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1122.20 | 1124.07 | 1122.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1111.80 | 1121.62 | 1121.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1111.80 | 1121.62 | 1121.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 1110.00 | 1119.29 | 1120.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 11:15:00 | 1107.20 | 1116.87 | 1119.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 1118.60 | 1115.06 | 1117.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 1118.60 | 1115.06 | 1117.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1110.90 | 1114.23 | 1117.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 1120.00 | 1114.23 | 1117.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1120.00 | 1115.38 | 1117.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1126.90 | 1115.38 | 1117.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1127.90 | 1117.89 | 1118.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1127.90 | 1117.89 | 1118.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1123.90 | 1119.09 | 1118.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 11:15:00 | 1133.30 | 1121.93 | 1120.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 10:15:00 | 1128.50 | 1128.66 | 1124.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 11:00:00 | 1128.50 | 1128.66 | 1124.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1126.70 | 1128.13 | 1125.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1126.70 | 1128.13 | 1125.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1129.20 | 1128.35 | 1126.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1129.90 | 1128.35 | 1126.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1126.40 | 1127.96 | 1126.13 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 1118.10 | 1124.18 | 1124.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 1117.20 | 1121.27 | 1122.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1125.00 | 1121.92 | 1122.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 1117.70 | 1122.70 | 1123.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1130.60 | 1124.28 | 1123.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 1130.60 | 1124.28 | 1123.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 15:15:00 | 1134.00 | 1126.37 | 1125.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1121.00 | 1125.30 | 1124.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:45:00 | 1112.00 | 1125.30 | 1124.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1120.00 | 1124.24 | 1124.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1114.50 | 1121.74 | 1123.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1114.00 | 1113.43 | 1117.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:45:00 | 1114.60 | 1113.43 | 1117.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1115.70 | 1113.88 | 1117.33 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1130.30 | 1118.68 | 1118.61 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1111.00 | 1117.10 | 1117.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 1106.50 | 1114.98 | 1116.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1114.50 | 1111.53 | 1114.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1114.50 | 1111.53 | 1114.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1116.00 | 1112.42 | 1114.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 1115.60 | 1112.42 | 1114.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1109.70 | 1111.88 | 1113.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1112.40 | 1111.88 | 1113.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1111.50 | 1111.77 | 1113.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:15:00 | 1110.00 | 1111.77 | 1113.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1110.00 | 1111.42 | 1113.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1114.50 | 1111.42 | 1113.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1121.00 | 1113.34 | 1113.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 1122.70 | 1113.34 | 1113.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1119.30 | 1114.53 | 1114.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 14:15:00 | 1123.80 | 1119.63 | 1118.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1118.90 | 1119.63 | 1118.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1118.90 | 1119.63 | 1118.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1119.60 | 1119.62 | 1118.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 1123.40 | 1120.38 | 1118.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1117.00 | 1120.16 | 1119.50 | SL hit (close<static) qty=1.00 sl=1118.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1132.50 | 1138.44 | 1138.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1130.60 | 1136.50 | 1137.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1136.00 | 1135.56 | 1136.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 1136.00 | 1135.56 | 1136.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1131.40 | 1134.73 | 1136.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 1130.80 | 1133.51 | 1135.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1126.10 | 1133.51 | 1135.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 13:15:00 | 1130.80 | 1126.49 | 1126.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 1130.80 | 1126.49 | 1126.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1133.60 | 1129.00 | 1127.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1124.90 | 1128.34 | 1127.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 1122.50 | 1128.34 | 1127.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1118.10 | 1126.29 | 1126.64 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1130.20 | 1127.07 | 1126.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 1138.00 | 1129.20 | 1127.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1126.30 | 1129.23 | 1128.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 1128.40 | 1129.23 | 1128.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1132.00 | 1129.78 | 1128.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 1126.10 | 1129.78 | 1128.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1128.40 | 1129.51 | 1128.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 1128.40 | 1129.51 | 1128.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1121.70 | 1127.94 | 1127.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1121.70 | 1127.94 | 1127.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1131.30 | 1128.62 | 1128.23 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1126.70 | 1128.01 | 1128.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 14:15:00 | 1118.50 | 1124.66 | 1126.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1011.00 | 1009.83 | 1036.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:45:00 | 1011.80 | 1009.83 | 1036.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 908.00 | 890.35 | 895.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 895.50 | 890.35 | 895.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 887.00 | 889.68 | 894.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 886.10 | 889.68 | 894.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 885.40 | 888.32 | 893.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 884.30 | 887.29 | 891.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 841.79 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 841.13 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 840.08 | 862.87 | 875.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 11:15:00 | 797.49 | 837.16 | 860.98 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 46 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 872.90 | 838.72 | 835.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 941.60 | 859.30 | 845.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 885.00 | 893.73 | 881.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 887.05 | 891.59 | 884.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 887.05 | 891.59 | 884.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 875.65 | 891.59 | 884.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 887.45 | 890.76 | 884.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 890.95 | 890.80 | 885.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 880.45 | 891.93 | 888.42 | SL hit (close<static) qty=1.00 sl=881.65 alert=retest2 |

### Cycle 47 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 857.75 | 881.06 | 884.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 10:15:00 | 849.75 | 874.79 | 880.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 894.45 | 874.02 | 879.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 894.45 | 874.02 | 879.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 877.10 | 874.64 | 879.02 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 889.65 | 881.32 | 881.01 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 878.60 | 880.77 | 880.79 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 882.00 | 881.02 | 880.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 886.75 | 882.26 | 881.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 866.70 | 881.67 | 881.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 866.70 | 881.67 | 881.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 857.40 | 876.81 | 879.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 852.60 | 871.97 | 876.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 876.30 | 872.30 | 876.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 876.30 | 872.30 | 876.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 889.45 | 875.73 | 877.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 889.45 | 875.73 | 877.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 881.40 | 876.86 | 877.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 897.55 | 876.86 | 877.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 903.20 | 882.13 | 880.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 910.50 | 887.80 | 882.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 1075.55 | 1083.50 | 1054.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 1075.55 | 1083.50 | 1054.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 1062.50 | 1070.30 | 1062.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 1062.50 | 1070.30 | 1062.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1060.55 | 1068.35 | 1062.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 1060.55 | 1068.35 | 1062.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1061.45 | 1066.97 | 1062.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 1060.20 | 1066.97 | 1062.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1057.00 | 1064.97 | 1061.88 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 1047.80 | 1057.95 | 1059.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 1042.75 | 1054.91 | 1057.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 948.40 | 943.22 | 960.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 948.40 | 943.22 | 960.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 954.00 | 946.67 | 953.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 937.45 | 946.67 | 953.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 939.25 | 945.19 | 952.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 932.25 | 942.45 | 950.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 973.40 | 947.20 | 950.43 | SL hit (close>static) qty=1.00 sl=957.30 alert=retest2 |

### Cycle 54 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 988.25 | 955.41 | 953.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 15:15:00 | 995.45 | 963.42 | 957.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 954.40 | 961.61 | 957.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 10:00:00 | 954.40 | 961.61 | 957.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 952.25 | 959.74 | 956.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 13:00:00 | 961.00 | 959.29 | 957.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:00:00 | 963.10 | 960.05 | 957.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:15:00 | 964.00 | 959.91 | 957.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 925.35 | 953.65 | 955.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 925.35 | 953.65 | 955.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 918.40 | 946.60 | 952.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 919.10 | 919.10 | 928.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 923.20 | 919.10 | 928.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 934.00 | 922.08 | 929.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:00:00 | 913.00 | 920.26 | 927.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 867.35 | 896.90 | 910.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 886.30 | 879.18 | 892.34 | SL hit (close>ema200) qty=0.50 sl=879.18 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 771.85 | 751.56 | 750.67 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 743.05 | 755.04 | 755.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 724.65 | 747.67 | 751.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 732.85 | 727.65 | 737.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 721.85 | 726.49 | 736.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 720.00 | 726.62 | 733.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 10:15:00 | 739.50 | 729.24 | 732.74 | SL hit (close>static) qty=1.00 sl=738.20 alert=retest2 |

### Cycle 58 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 748.30 | 736.48 | 735.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 750.35 | 739.26 | 736.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 778.65 | 780.46 | 767.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:45:00 | 783.00 | 780.46 | 767.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 824.00 | 826.79 | 819.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:45:00 | 818.60 | 826.79 | 819.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 818.00 | 825.03 | 819.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 794.75 | 825.03 | 819.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 789.10 | 817.84 | 816.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 786.95 | 817.84 | 816.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 789.95 | 812.26 | 814.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 785.85 | 803.04 | 809.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 805.60 | 802.45 | 806.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 800.05 | 802.77 | 806.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:15:00 | 800.30 | 802.77 | 806.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:00:00 | 799.75 | 802.71 | 805.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 801.80 | 802.88 | 805.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 803.55 | 803.01 | 804.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:30:00 | 804.40 | 803.01 | 804.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 804.30 | 803.27 | 804.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:30:00 | 805.35 | 803.27 | 804.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 800.55 | 802.73 | 804.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 808.30 | 805.38 | 805.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 808.30 | 805.38 | 805.26 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 804.00 | 804.97 | 805.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 14:15:00 | 801.70 | 803.97 | 804.57 | Break + close below crossover candle low |

### Cycle 62 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 830.50 | 809.10 | 806.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 864.60 | 830.26 | 819.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 15:15:00 | 859.55 | 860.12 | 849.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 856.80 | 860.12 | 849.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 855.85 | 860.28 | 854.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 853.40 | 860.28 | 854.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 852.00 | 858.62 | 853.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 855.35 | 858.62 | 853.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 846.85 | 856.27 | 853.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 841.85 | 856.27 | 853.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 831.00 | 851.21 | 851.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 829.30 | 844.16 | 847.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 863.80 | 845.23 | 846.80 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 867.00 | 849.58 | 848.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 884.20 | 868.62 | 864.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 867.25 | 875.17 | 870.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 867.25 | 875.17 | 870.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 868.60 | 873.85 | 870.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 865.65 | 873.85 | 870.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 868.85 | 872.85 | 870.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:00:00 | 869.65 | 870.80 | 870.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 874.00 | 870.04 | 869.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 15:15:00 | 1173.00 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.56% |
| BUY | retest2 | 2025-05-15 09:30:00 | 1176.00 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.30% |
| BUY | retest2 | 2025-05-15 14:15:00 | 1173.50 | 2025-05-21 12:15:00 | 1203.00 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-05-26 13:30:00 | 1190.20 | 2025-05-28 09:15:00 | 1226.70 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-06-05 11:15:00 | 1239.40 | 2025-06-12 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2025-06-05 11:45:00 | 1239.20 | 2025-06-12 15:15:00 | 1275.00 | STOP_HIT | 1.00 | 2.89% |
| SELL | retest2 | 2025-06-13 15:15:00 | 1282.00 | 2025-06-16 12:15:00 | 1291.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-06-18 12:45:00 | 1272.80 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1272.30 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2025-06-19 09:45:00 | 1272.80 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-19 11:30:00 | 1266.70 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2025-06-20 09:15:00 | 1270.00 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2025-06-20 09:45:00 | 1268.40 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-06-20 11:30:00 | 1269.60 | 2025-06-25 15:15:00 | 1252.00 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-06-27 12:30:00 | 1260.00 | 2025-06-27 15:15:00 | 1240.20 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-02 14:45:00 | 1245.00 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-03 09:30:00 | 1246.50 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-03 10:15:00 | 1246.50 | 2025-07-03 11:15:00 | 1237.20 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-08 14:15:00 | 1241.40 | 2025-07-09 09:15:00 | 1255.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-07-08 15:00:00 | 1242.40 | 2025-07-09 09:15:00 | 1255.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-11 12:30:00 | 1256.10 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-11 14:45:00 | 1256.50 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-11 15:15:00 | 1260.00 | 2025-07-14 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-17 14:30:00 | 1242.50 | 2025-07-25 11:15:00 | 1180.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1242.10 | 2025-07-25 11:15:00 | 1179.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1241.30 | 2025-07-25 11:15:00 | 1179.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1239.80 | 2025-07-25 14:15:00 | 1177.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1239.30 | 2025-07-25 14:15:00 | 1177.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 15:15:00 | 1234.90 | 2025-07-25 14:15:00 | 1173.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 14:30:00 | 1242.50 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.63% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1242.10 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.60% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1241.30 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.54% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1239.80 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.43% |
| SELL | retest2 | 2025-07-21 14:15:00 | 1239.30 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.39% |
| SELL | retest2 | 2025-07-21 15:15:00 | 1234.90 | 2025-07-30 10:15:00 | 1135.30 | STOP_HIT | 0.50 | 8.07% |
| SELL | retest2 | 2025-08-08 14:00:00 | 1106.20 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-08-12 15:15:00 | 1106.00 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-08-13 09:30:00 | 1107.90 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-08-14 09:45:00 | 1107.10 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-18 11:15:00 | 1104.20 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-18 15:00:00 | 1102.80 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-08-19 10:15:00 | 1101.90 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-08-19 13:00:00 | 1103.00 | 2025-08-20 13:15:00 | 1108.90 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-08-22 15:00:00 | 1126.40 | 2025-08-26 11:15:00 | 1119.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-08-25 09:30:00 | 1132.40 | 2025-08-26 11:15:00 | 1119.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-09 10:30:00 | 1115.10 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-09-10 11:00:00 | 1121.80 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-12 12:15:00 | 1117.40 | 2025-09-12 13:15:00 | 1112.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1149.00 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-09-17 10:15:00 | 1144.20 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-17 11:00:00 | 1143.30 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-17 12:00:00 | 1143.20 | 2025-09-17 15:15:00 | 1131.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-18 13:15:00 | 1144.50 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-09-18 14:45:00 | 1154.30 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-18 15:15:00 | 1150.00 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-19 11:00:00 | 1145.90 | 2025-09-19 12:15:00 | 1122.70 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-24 10:30:00 | 1101.40 | 2025-10-01 10:15:00 | 1046.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:30:00 | 1101.40 | 2025-10-03 09:15:00 | 1052.60 | STOP_HIT | 0.50 | 4.43% |
| BUY | retest2 | 2025-10-24 09:45:00 | 1078.00 | 2025-10-31 14:15:00 | 1082.70 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-11-12 10:30:00 | 1033.70 | 2025-11-12 13:15:00 | 1060.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-11-19 11:45:00 | 1111.10 | 2025-11-24 14:15:00 | 1110.30 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-05 09:30:00 | 1117.70 | 2025-12-05 10:15:00 | 1130.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-16 12:00:00 | 1123.40 | 2025-12-17 09:15:00 | 1117.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 11:00:00 | 1123.90 | 2025-12-26 11:15:00 | 1132.50 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-12-30 10:30:00 | 1130.80 | 2026-01-01 13:15:00 | 1130.80 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-30 11:15:00 | 1126.10 | 2026-01-01 13:15:00 | 1130.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-01-22 10:15:00 | 886.10 | 2026-01-27 09:15:00 | 841.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 885.40 | 2026-01-27 09:15:00 | 841.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 884.30 | 2026-01-27 09:15:00 | 840.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 886.10 | 2026-01-27 11:15:00 | 797.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 885.40 | 2026-01-27 11:15:00 | 796.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 15:15:00 | 884.30 | 2026-01-27 11:15:00 | 795.87 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-02 15:00:00 | 890.95 | 2026-02-03 12:15:00 | 880.45 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-02-03 14:15:00 | 891.00 | 2026-02-03 14:15:00 | 876.85 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-02-27 10:45:00 | 932.25 | 2026-02-27 13:15:00 | 973.40 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2026-03-02 13:00:00 | 961.00 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-03-02 14:00:00 | 963.10 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-03-02 15:15:00 | 964.00 | 2026-03-04 09:15:00 | 925.35 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-03-06 10:00:00 | 913.00 | 2026-03-09 09:15:00 | 867.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:00:00 | 913.00 | 2026-03-10 09:15:00 | 886.30 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-04-01 11:00:00 | 721.85 | 2026-04-02 10:15:00 | 739.50 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2026-04-01 14:45:00 | 720.00 | 2026-04-02 10:15:00 | 739.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-15 11:45:00 | 800.05 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-15 12:15:00 | 800.30 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-04-15 15:00:00 | 799.75 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-16 09:30:00 | 801.80 | 2026-04-17 09:15:00 | 808.30 | STOP_HIT | 1.00 | -0.81% |
