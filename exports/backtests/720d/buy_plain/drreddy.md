# DRREDDY (DRREDDY)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1306.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 23 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 0 / 18 / 0
- **Avg / median % per leg:** -1.40% / -1.87%
- **Sum % (uncompounded):** -25.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 0 | 18 | 0 | -1.40% | -25.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.03% | -8.1% |
| BUY @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 0 | 16 | 0 | -1.07% | -17.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.03% | -8.1% |
| retest2 (combined) | 16 | 6 | 37.5% | 0 | 16 | 0 | -1.07% | -17.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 13:15:00 | 1397.25 | 1282.56 | 1282.55 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 1232.10 | 1182.48 | 1182.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.73 | 1183.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.54 | 1259.40 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-04 14:15:00 | 1307.30 | 1291.33 | 1262.97 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 15:15:00 | 1311.70 | 1291.54 | 1263.21 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 1307.40 | 1291.77 | 1263.61 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-07 11:15:00 | 1300.40 | 1291.85 | 1263.79 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-07 12:15:00 | 1306.70 | 1292.00 | 1264.01 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 13:15:00 | 1307.80 | 1292.16 | 1264.22 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | SL hit (close<ema400) qty=1.00 sl=1265.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | SL hit (close<ema400) qty=1.00 sl=1265.49 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-10 12:15:00 | 1269.00 | 1289.40 | 1265.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-10 13:15:00 | 1267.20 | 1289.18 | 1265.47 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-17 12:15:00 | 1270.40 | 1280.02 | 1264.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:15:00 | 1268.40 | 1279.90 | 1264.22 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1251.90 | 1279.13 | 1264.15 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-24 09:15:00 | 1276.90 | 1272.11 | 1262.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1282.20 | 1272.21 | 1262.41 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-24 15:15:00 | 1268.90 | 1272.25 | 1262.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1277.10 | 1272.30 | 1262.75 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-07-31 10:15:00 | 1277.40 | 1275.40 | 1265.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1279.70 | 1275.44 | 1265.76 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1243.40 | 1275.07 | 1265.81 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1243.40 | 1275.07 | 1265.81 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1243.40 | 1275.07 | 1265.81 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1243.40 | 1275.07 | 1265.81 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-14 14:15:00 | 1261.50 | 1248.55 | 1252.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-14 15:15:00 | 1251.00 | 1248.57 | 1252.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-18 13:15:00 | 1262.40 | 1248.84 | 1252.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 1262.00 | 1248.97 | 1252.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-21 10:15:00 | 1261.00 | 1249.13 | 1252.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:15:00 | 1269.70 | 1249.34 | 1252.62 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-28 14:15:00 | 1260.10 | 1255.12 | 1255.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 1263.80 | 1255.21 | 1255.36 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-29 10:15:00 | 1260.30 | 1255.19 | 1255.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 1259.70 | 1255.23 | 1255.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1255.60 | 1255.34 | 1255.42 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.70 | 1255.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-04 12:15:00 | 1269.90 | 1257.87 | 1256.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-04 13:15:00 | 1258.60 | 1257.88 | 1256.78 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-05 14:15:00 | 1267.90 | 1258.20 | 1256.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1269.70 | 1258.32 | 1257.05 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1250.80 | 1258.30 | 1257.08 | SL hit (close<static) qty=1.00 sl=1254.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 09:15:00 | 1273.10 | 1258.37 | 1257.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1276.80 | 1258.56 | 1257.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 1252.90 | 1283.73 | 1273.64 | SL hit (close<static) qty=1.00 sl=1254.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1277.80 | 1259.86 | 1262.63 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 1288.80 | 1265.08 | 1265.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.08 | 1265.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1293.90 | 1266.73 | 1265.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA400 retest candle locked |

### Cycle 5 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.00 | 1252.56 | 1252.55 | EMA200 above EMA400 |

### Cycle 6 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1259.60 | 1252.60 | 1252.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1261.40 | 1252.69 | 1252.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA400 retest candle locked |

### Cycle 7 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1268.00 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.46 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 1284.00 | 1282.93 | 1268.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1284.80 | 1282.95 | 1268.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1292.50 | 1283.04 | 1269.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1298.40 | 1283.19 | 1269.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1298.00 | 1283.18 | 1270.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1294.60 | 1283.29 | 1270.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.93 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.93 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.93 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1291.50 | 1281.29 | 1270.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1291.90 | 1281.39 | 1270.55 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1260.40 | 1282.82 | 1271.98 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1260.40 | 1282.82 | 1271.98 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1306.10 | 1243.15 | 1251.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1299.30 | 1243.71 | 1251.49 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.44 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.00 | 1274.75 | 1267.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1274.84 | 1267.67 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-04 15:15:00 | 1311.70 | 2025-07-10 09:15:00 | 1256.90 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest1 | 2025-07-07 13:15:00 | 1307.80 | 2025-07-10 09:15:00 | 1256.90 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-07-17 13:15:00 | 1268.40 | 2025-07-18 10:15:00 | 1251.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-24 10:15:00 | 1282.20 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1277.10 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1279.70 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-08-18 14:15:00 | 1262.00 | 2025-09-01 09:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2025-08-21 11:15:00 | 1269.70 | 2025-09-01 09:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-08-28 15:15:00 | 1263.80 | 2025-09-01 09:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-08-29 11:15:00 | 1259.70 | 2025-09-01 09:15:00 | 1272.80 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1269.70 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1276.80 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-20 10:15:00 | 1277.80 | 2025-10-27 13:15:00 | 1288.80 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2026-03-17 14:15:00 | 1284.80 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1298.40 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1294.60 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1291.90 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-04-23 10:15:00 | 1299.30 | 2026-04-28 09:15:00 | 1340.40 | STOP_HIT | 1.00 | 3.16% |
