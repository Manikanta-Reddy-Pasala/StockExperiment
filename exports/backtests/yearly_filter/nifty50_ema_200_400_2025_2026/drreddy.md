# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 5 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 31
- **Target hits / Stop hits / Partials:** 0 / 34 / 3
- **Avg / median % per leg:** -1.08% / -1.10%
- **Sum % (uncompounded):** -39.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 0 | 0.0% | 0 | 27 | 0 | -1.94% | -52.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 0 | 0.0% | 0 | 27 | 0 | -1.94% | -52.5% |
| SELL (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.27% | 12.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.27% | 12.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 6 | 16.2% | 0 | 34 | 3 | -1.08% | -39.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 1232.10 | 1182.48 | 1182.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.73 | 1183.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.54 | 1259.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 1278.40 | 1294.54 | 1259.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1256.90 | 1290.18 | 1265.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1257.30 | 1289.85 | 1265.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1253.90 | 1289.85 | 1265.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1269.00 | 1289.40 | 1265.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 1272.10 | 1280.11 | 1264.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 1263.50 | 1279.74 | 1264.22 | SL hit (close<static) qty=1.00 sl=1263.90 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.75 | 1257.90 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.70 | 1255.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1256.70 | 1257.31 | 1256.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1250.00 | 1257.23 | 1256.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1252.80 | 1257.23 | 1256.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1257.60 | 1257.24 | 1256.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:30:00 | 1264.70 | 1257.30 | 1256.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 12:00:00 | 1262.10 | 1257.30 | 1256.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:00:00 | 1262.00 | 1257.47 | 1256.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1263.30 | 1257.88 | 1256.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1257.50 | 1258.38 | 1257.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 1257.50 | 1258.38 | 1257.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1256.10 | 1258.36 | 1257.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:45:00 | 1254.20 | 1258.36 | 1257.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1259.40 | 1258.37 | 1257.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 1261.10 | 1258.37 | 1257.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:45:00 | 1261.40 | 1258.37 | 1257.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1250.80 | 1258.30 | 1257.08 | SL hit (close<static) qty=1.00 sl=1254.20 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.94 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.10 | 1265.79 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 1272.00 | 1259.68 | 1262.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1277.80 | 1259.86 | 1262.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 1277.80 | 1259.86 | 1262.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1280.90 | 1264.32 | 1264.69 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.08 | 1265.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1293.90 | 1266.73 | 1265.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 1252.00 | 1267.38 | 1266.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1254.70 | 1267.25 | 1266.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 1243.50 | 1267.25 | 1266.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.09 | 1265.13 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.00 | 1252.56 | 1252.55 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1247.10 | 1252.53 | 1252.54 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1259.60 | 1252.60 | 1252.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1261.40 | 1252.69 | 1252.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1259.50 | 1263.96 | 1259.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1260.10 | 1263.92 | 1259.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1265.30 | 1263.94 | 1259.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 1265.80 | 1263.92 | 1259.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:30:00 | 1266.70 | 1263.94 | 1259.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1267.20 | 1263.98 | 1259.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1265.10 | 1264.36 | 1259.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 1270.20 | 1264.38 | 1259.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 1269.90 | 1264.44 | 1260.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 1254.10 | 1264.54 | 1260.18 | SL hit (close<static) qty=1.00 sl=1255.50 alert=retest2 |

### Cycle 10 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.67 | 1256.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 1243.50 | 1225.87 | 1239.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1243.00 | 1226.04 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1246.00 | 1226.04 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1247.70 | 1226.38 | 1239.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1247.70 | 1226.38 | 1239.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1232.00 | 1226.54 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1238.70 | 1226.54 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1238.40 | 1226.66 | 1239.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1242.90 | 1226.66 | 1239.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1245.30 | 1226.85 | 1239.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:30:00 | 1235.70 | 1227.32 | 1239.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 1235.10 | 1227.32 | 1239.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:30:00 | 1231.60 | 1227.57 | 1239.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.91 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1173.34 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 15:15:00 | 1170.02 | 1224.08 | 1235.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1221.90 | 1221.12 | 1233.83 | SL hit (close>ema200) qty=0.50 sl=1221.12 alert=retest2 |

### Cycle 11 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1268.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 13:00:00 | 1283.30 | 1283.55 | 1268.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 1269.00 | 1283.70 | 1268.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 1271.30 | 1283.58 | 1268.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:45:00 | 1267.10 | 1283.58 | 1268.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1268.20 | 1283.43 | 1268.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1268.20 | 1283.43 | 1268.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1273.50 | 1283.33 | 1268.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 1271.20 | 1283.33 | 1268.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1268.40 | 1282.96 | 1268.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:45:00 | 1267.70 | 1282.96 | 1268.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1279.00 | 1282.92 | 1268.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:30:00 | 1268.10 | 1282.92 | 1268.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1276.30 | 1283.64 | 1269.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1298.00 | 1283.03 | 1269.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.93 | SL hit (close<static) qty=1.00 sl=1266.70 alert=retest2 |

### Cycle 12 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.25 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:30:00 | 1272.10 | 1274.75 | 1267.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-17 12:15:00 | 1272.10 | 2025-07-17 14:15:00 | 1263.50 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1284.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-07-24 10:00:00 | 1276.90 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1283.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-28 14:30:00 | 1289.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-07-29 09:45:00 | 1288.40 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-07-29 10:30:00 | 1288.20 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2025-07-29 12:00:00 | 1290.10 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-07-31 11:30:00 | 1280.50 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-31 12:45:00 | 1280.00 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-09-03 11:30:00 | 1264.70 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-03 12:00:00 | 1262.10 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-03 15:00:00 | 1262.00 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1263.30 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-09-08 13:15:00 | 1261.10 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-08 13:45:00 | 1261.40 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-09 09:15:00 | 1274.80 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-26 09:30:00 | 1267.40 | 2025-09-29 11:15:00 | 1244.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-10 14:30:00 | 1264.10 | 2025-10-13 09:15:00 | 1251.50 | STOP_HIT | 1.00 | -1.00% |
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
