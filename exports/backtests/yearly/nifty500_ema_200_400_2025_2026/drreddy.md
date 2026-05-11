# Dr. Reddy's Laboratories Ltd. (DRREDDY)

## Backtest Summary

- **Window:** 2025-07-25 09:15:00 → 2026-05-08 15:15:00 (1346 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 0 / 15 / 3
- **Avg / median % per leg:** 0.05% / -0.89%
- **Sum % (uncompounded):** 0.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.46% | -11.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.46% | -11.7% |
| SELL (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.27% | 12.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.27% | 12.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 6 | 33.3% | 0 | 15 | 3 | 0.05% | 1.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1283.10 | 1264.66 | 1264.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1290.10 | 1265.89 | 1265.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1252.00 | 1266.85 | 1265.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1252.00 | 1266.85 | 1265.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1252.00 | 1266.85 | 1265.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 1252.00 | 1266.85 | 1265.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1254.70 | 1266.73 | 1265.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 1243.50 | 1266.73 | 1265.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1264.59 | 1264.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 14:15:00 | 1265.80 | 1252.34 | 1252.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 1269.50 | 1252.79 | 1252.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.93 | 1258.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.50 | 1263.93 | 1258.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.50 | 1263.93 | 1258.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1259.50 | 1263.93 | 1258.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1260.10 | 1263.89 | 1258.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1265.30 | 1263.90 | 1259.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 1265.80 | 1263.89 | 1259.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 1266.70 | 1263.90 | 1259.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1267.20 | 1263.94 | 1259.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1265.10 | 1264.33 | 1259.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 1270.20 | 1264.36 | 1259.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 1269.90 | 1264.41 | 1259.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1254.10 | 1264.52 | 1260.07 | SL hit (close<static) qty=1.00 sl=1255.50 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.65 | 1256.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.38 | 1255.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.86 | 1239.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.86 | 1239.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.86 | 1239.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 1243.50 | 1225.86 | 1239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1243.00 | 1226.03 | 1239.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1246.00 | 1226.03 | 1239.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1247.70 | 1226.37 | 1239.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1247.70 | 1226.37 | 1239.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1232.00 | 1226.54 | 1239.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1238.70 | 1226.54 | 1239.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1238.40 | 1226.65 | 1239.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1242.90 | 1226.65 | 1239.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1245.30 | 1226.84 | 1239.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 1235.70 | 1227.31 | 1239.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 1235.10 | 1227.31 | 1239.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1231.60 | 1227.56 | 1239.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.91 | 1224.07 | 1235.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.34 | 1224.07 | 1235.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1170.02 | 1224.07 | 1235.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1221.90 | 1221.11 | 1233.78 | SL hit (close>ema200) qty=0.50 sl=1221.11 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.90 | 1241.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1267.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 1283.30 | 1283.55 | 1267.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 1269.00 | 1283.70 | 1268.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 1271.30 | 1283.58 | 1268.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:45:00 | 1267.10 | 1283.58 | 1268.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1268.20 | 1283.43 | 1268.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1268.20 | 1283.43 | 1268.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1273.50 | 1283.33 | 1268.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 1271.20 | 1283.33 | 1268.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1268.40 | 1282.96 | 1268.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:45:00 | 1267.70 | 1282.96 | 1268.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1279.00 | 1282.92 | 1268.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:30:00 | 1268.10 | 1282.92 | 1268.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1276.30 | 1283.63 | 1269.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1298.00 | 1283.03 | 1269.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.91 | SL hit (close<static) qty=1.00 sl=1266.70 alert=retest2 |

### Cycle 6 — SELL (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 10:15:00 | 1190.40 | 1262.52 | 1262.87 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:30:00 | 1272.10 | 1274.75 | 1267.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-24 12:00:00 | 1265.30 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-24 12:45:00 | 1265.80 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-24 14:30:00 | 1266.70 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1267.20 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-31 12:00:00 | 1270.20 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-31 12:45:00 | 1269.90 | 2026-01-01 09:15:00 | 1254.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-27 13:30:00 | 1235.70 | 2026-02-01 15:15:00 | 1173.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 14:15:00 | 1235.10 | 2026-02-01 15:15:00 | 1173.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1231.60 | 2026-02-01 15:15:00 | 1170.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-27 13:30:00 | 1235.70 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2026-01-27 14:15:00 | 1235.10 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-01-28 09:30:00 | 1231.60 | 2026-02-03 09:15:00 | 1221.90 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2026-02-03 12:15:00 | 1235.20 | 2026-02-04 09:15:00 | 1243.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-02-03 15:15:00 | 1228.90 | 2026-02-06 14:15:00 | 1240.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1230.60 | 2026-02-06 14:15:00 | 1240.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-06 12:15:00 | 1229.40 | 2026-02-09 09:15:00 | 1263.70 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1298.00 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-03-25 09:45:00 | 1287.80 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.13% |
